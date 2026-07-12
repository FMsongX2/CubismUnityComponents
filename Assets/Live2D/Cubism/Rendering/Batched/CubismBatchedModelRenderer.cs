/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */


using Live2D.Cubism.Core;
using Live2D.Cubism.Rendering.Util;
using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;


namespace Live2D.Cubism.Rendering
{
    /// <summary>
    /// Renders a whole model through a single dynamic mesh with one draw call per
    /// state batch instead of one draw call per drawable, and renders all clipping
    /// masks into a tiled atlas once per frame instead of re-rendering them per
    /// masked drawable. Used by the mobile fast path when
    /// <see cref="CubismRenderController.IsBatchedRenderingActive"/> is set.
    /// </summary>
    public sealed class CubismBatchedModelRenderer : IDisposable
    {
        /// <summary>
        /// Maximum mask groups per model (limited by the shader parameter arrays;
        /// slot 0 is reserved for "not masked").
        /// </summary>
        public const int MaxMaskGroups = 64;

        /// <summary>
        /// Resolution of the mask atlas render texture.
        /// </summary>
        public static int MaskAtlasSize = 1024;


        #region Static Shader Property IDs

        private static readonly int MainTextureId = Shader.PropertyToID("_MainTex");
        private static readonly int MainTextureArrayId = Shader.PropertyToID("_MainTexArray");
        private static readonly int ModelOpacityId = Shader.PropertyToID("cubism_ModelOpacity");
        private static readonly int MaskTextureId = Shader.PropertyToID("cubism_MaskTexture");
        private static readonly int MaskTileId = Shader.PropertyToID("cubism_MaskTile");
        private static readonly int MaskTransformId = Shader.PropertyToID("cubism_MaskTransform");
        private static readonly int MaskTilesArrayId = Shader.PropertyToID("_CubismMaskTiles");
        private static readonly int MaskTransformsArrayId = Shader.PropertyToID("_CubismMaskTransforms");
        private static readonly int SrcColorId = Shader.PropertyToID("_SrcColor");
        private static readonly int DstColorId = Shader.PropertyToID("_DstColor");
        private static readonly int SrcAlphaId = Shader.PropertyToID("_SrcAlpha");
        private static readonly int DstAlphaId = Shader.PropertyToID("_DstAlpha");
        private static readonly int CullId = Shader.PropertyToID("_Cull");

        #endregion


        #region Types

        /// <summary>
        /// Interleaved layout of vertex stream 1 (all low-frequency per-vertex data).
        /// Field order must match the vertex attribute declaration (Color, TexCoord1, TexCoord2).
        /// </summary>
        private struct Stream1Data
        {
            /// <summary>Tint color; alpha premultiplied with drawable opacity.</summary>
            public Color32 Color;

            /// <summary>Multiply color in rgb, mask group index in a.</summary>
            public Color32 MultiplyAndGroup;

            /// <summary>Screen color in rgb, mask invert flag in a.</summary>
            public Color32 ScreenAndInvert;
        }

        /// <summary>
        /// A run of consecutive drawables sharing render state; drawn with one draw call.
        /// </summary>
        private struct Batch
        {
            public int TextureSlot;
            public BlendTypes.ColorBlend ColorBlend;
            public bool IsDoubleSided;
            public int IndexStart;
            public int IndexCount;
        }

        /// <summary>
        /// Static per-frame mask atlas draw (all mask meshes of one group sharing one texture).
        /// </summary>
        private struct MaskSection
        {
            public int GroupIndex;
            public int SubMeshIndex;
            public Material Material;
            public MaterialPropertyBlock Properties;
        }

        #endregion


        #region Fields

        private CubismRenderController _controller;
        private CubismRenderer[] _renderersByDrawable;

        private int _drawableCount;
        private int _totalVertexCount;
        private int _totalIndexCount;
        private int _maskIndexCount;

        // Per-drawable static tables (indexed by unmanaged drawable index).
        private int[] _vertexBase;
        private int[] _vertexCount;
        private int[] _indexBase;
        private int[] _indexCount;
        private BlendTypes.ColorBlend[] _colorBlend;
        private bool[] _isDoubleSided;
        private byte[] _maskGroup;
        private bool[] _isInverted;
        private int[] _textureSlot;

        // Per-drawable dynamic state.
        private float[] _opacities;
        private bool[] _visible;
        private int[] _renderOrders;
        private int[] _orderedDrawables;

        // Vertex/index storage.
        private NativeArray<Vector3> _positions;
        private NativeArray<Stream1Data> _stream1;
        private NativeArray<Vector3> _uvs;
        private NativeArray<ushort> _bakedIndices16;
        private NativeArray<uint> _bakedIndices32;
        private NativeArray<ushort> _indexBuffer16;
        private NativeArray<uint> _indexBuffer32;
        private bool _use32BitIndices;

        private Mesh _mesh;

        // Batching.
        private readonly List<Batch> _batches = new List<Batch>(32);
        private int _mainIndexCount;

        // Textures / materials.
        private Texture[] _textures;
        private Texture2DArray _textureArray;
        private bool _useTextureArray;
        private readonly Dictionary<long, Material> _materials = new Dictionary<long, Material>();
        private MaterialPropertyBlock _modelProperties;

        // Masks.
        private int _maskGroupCount;
        private int[][] _maskGroupMembers;
        private Vector4[] _maskTiles;
        private Vector4[] _maskTransforms;
        private readonly List<MaskSection> _maskSections = new List<MaskSection>(32);
        private readonly List<SubMeshDescriptor> _maskSubMeshes = new List<SubMeshDescriptor>(32);
        private RenderTexture _maskAtlas;

        // Scratch list for SetSubMeshes (mask sections + batches).
        private readonly List<SubMeshDescriptor> _subMeshScratch = new List<SubMeshDescriptor>(64);

        // Dirty flags.
        private bool _positionsDirty;
        private bool _stream1Dirty;
        private bool _indicesDirty;
        private bool _texturesDirty;
        private bool _receivedFirstData;
        private int _lastFlushedFrame = -1;
        private int _lastMaskUpdateFrame = -1;

        private bool _isDisposed;
        private bool _isBroken;

        /// <summary>
        /// In linear color space the legacy path converts multiply/screen colors from
        /// sRGB to linear via <see cref="MaterialPropertyBlock.SetColor"/>; vertex
        /// attributes carry raw values, so the conversion happens on the CPU instead.
        /// (Vertex tint colors are raw in the legacy path too and stay unconverted.)
        /// </summary>
        private bool _convertBlendColorsToLinear;

        #endregion


        /// <summary>
        /// True when initialization succeeded and the renderer can record draws.
        /// </summary>
        public bool IsValid
        {
            get { return !_isDisposed && !_isBroken && _mesh != null; }
        }


        #region Initialization

        /// <summary>
        /// Checks whether a model qualifies for the batched fast path. Must not touch
        /// <see cref="CubismRenderController.Renderers"/> as it runs before renderer
        /// initialization; renderer-level conditions are validated separately in
        /// <see cref="AreRenderersEligible"/>.
        /// </summary>
        public static bool IsModelEligible(CubismRenderController controller)
        {
            var model = controller.Model;

            if (model == null || model.Drawables == null || model.Drawables.Length < 1)
            {
                return false;
            }

            // Blend color handlers receive per-drawable change events from the legacy path.
            if (controller.MultiplyColorHandler != null || controller.ScreenColorHandler != null)
            {
                return false;
            }

            // Parts offscreens require the buffered legacy pipeline.
            if (model.Offscreens != null && model.Offscreens.Length > 0)
            {
                return false;
            }

            // Only sorting modes whose draw sequence equals the core render order.
            if (controller.SortingMode != CubismSortingMode.BackToFrontZ
                && controller.SortingMode != CubismSortingMode.BackToFrontOrder)
            {
                return false;
            }

            var maskGroupKeys = new HashSet<string>();
            var drawables = model.Drawables;

            for (var i = 0; i < drawables.Length; i++)
            {
                var drawable = drawables[i];

                // Only hardware-expressible blend modes qualify.
                switch (drawable.ColorBlend)
                {
                    case BlendTypes.ColorBlend.Normal:
                        if (drawable.AlphaBlend != BlendTypes.AlphaBlend.Over)
                        {
                            return false;
                        }
                        break;
                    case BlendTypes.ColorBlend.Add:
                    case BlendTypes.ColorBlend.Multiply:
                        break;
                    default:
                        return false;
                }

                if (drawable.IsMasked)
                {
                    maskGroupKeys.Add(MaskGroupKey(drawable.Masks));
                }
            }

            // Mask groups must fit the shader parameter arrays (slot 0 is reserved).
            if (maskGroupKeys.Count > MaxMaskGroups - 1)
            {
                return false;
            }

            return true;
        }


        /// <summary>
        /// Renderer-level eligibility, checked after renderers are initialized.
        /// </summary>
        public static bool AreRenderersEligible(CubismRenderController controller)
        {
            // Per-drawable local sorting orders would reorder drawables away from
            // the core render order.
            var renderers = controller.Renderers;

            for (var i = 0; i < renderers.Length; i++)
            {
                if (renderers[i] != null && renderers[i].LocalSortingOrder != 0)
                {
                    return false;
                }
            }

            return true;
        }

        private static string MaskGroupKey(CubismDrawable[] masks)
        {
            var indices = new int[masks.Length];
            for (var i = 0; i < masks.Length; i++)
            {
                indices[i] = masks[i] != null ? masks[i].UnmanagedIndex : -1;
            }
            Array.Sort(indices);
            return string.Join(",", indices);
        }


        /// <summary>
        /// Builds all static tables, the mesh, materials, and the mask atlas.
        /// </summary>
        public CubismBatchedModelRenderer(CubismRenderController controller)
        {
            _controller = controller;

            try
            {
                Build();
            }
            catch (Exception e)
            {
                Debug.LogError($"[CubismBatchedModelRenderer] Initialization failed, falling back to legacy rendering: {e}");
                _isBroken = true;
                DisposeResources();
            }
        }

        private void Build()
        {
            var model = _controller.Model;
            var drawables = model.Drawables;
            _drawableCount = drawables.Length;

            _convertBlendColorsToLinear = QualitySettings.activeColorSpace == ColorSpace.Linear;

            // Map renderers by unmanaged drawable index.
            var drawableRenderers = _controller.DrawableRenderers;
            _renderersByDrawable = new CubismRenderer[_drawableCount];
            for (var i = 0; i < drawableRenderers.Length; i++)
            {
                _renderersByDrawable[drawableRenderers[i].Drawable.UnmanagedIndex] = drawableRenderers[i];

                // Editor-only: scene-picking MeshFilters persisted from edit mode would
                // render their stale meshes through the regular pipeline during play.
                var meshFilter = drawableRenderers[i].GetComponent<MeshFilter>();
                if (meshFilter != null)
                {
                    meshFilter.sharedMesh = null;
                }
            }

            // Static per-drawable tables.
            _vertexBase = new int[_drawableCount];
            _vertexCount = new int[_drawableCount];
            _indexBase = new int[_drawableCount];
            _indexCount = new int[_drawableCount];
            _colorBlend = new BlendTypes.ColorBlend[_drawableCount];
            _isDoubleSided = new bool[_drawableCount];
            _maskGroup = new byte[_drawableCount];
            _isInverted = new bool[_drawableCount];
            _textureSlot = new int[_drawableCount];
            _opacities = new float[_drawableCount];
            _visible = new bool[_drawableCount];
            _renderOrders = new int[_drawableCount];
            _orderedDrawables = new int[_drawableCount];

            _totalVertexCount = 0;
            _totalIndexCount = 0;

            var vertexUvs = new Vector2[_drawableCount][];
            var localIndices = new int[_drawableCount][];
            var initialPositions = new Vector3[_drawableCount][];
            var renderOrders = model.AllDrawObjectsRenderOrder;

            for (var i = 0; i < _drawableCount; i++)
            {
                var drawable = drawables[i];
                var unmanagedIndex = drawable.UnmanagedIndex;

                vertexUvs[unmanagedIndex] = drawable.VertexUvs;
                localIndices[unmanagedIndex] = drawable.Indices;
                initialPositions[unmanagedIndex] = drawable.VertexPositions;

                _vertexBase[unmanagedIndex] = 0; // Filled below in index order.
                _vertexCount[unmanagedIndex] = vertexUvs[unmanagedIndex].Length;
                _indexCount[unmanagedIndex] = localIndices[unmanagedIndex].Length;
                _colorBlend[unmanagedIndex] = drawable.ColorBlend;
                _isDoubleSided[unmanagedIndex] = drawable.IsDoubleSided;
                _isInverted[unmanagedIndex] = drawable.IsInverted;
                _renderOrders[unmanagedIndex] = renderOrders[unmanagedIndex];
                _opacities[unmanagedIndex] = 1.0f;
                _visible[unmanagedIndex] = true;
            }

            for (var i = 0; i < _drawableCount; i++)
            {
                _vertexBase[i] = _totalVertexCount;
                _indexBase[i] = _totalIndexCount;
                _totalVertexCount += _vertexCount[i];
                _totalIndexCount += _indexCount[i];
            }

            _use32BitIndices = _totalVertexCount > ushort.MaxValue;

            // Mask groups.
            BuildMaskGroups(drawables);

            // Texture table + materials (also fills _textureSlot).
            BuildTextures();

            // Vertex/index storage.
            _positions = new NativeArray<Vector3>(_totalVertexCount, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            _stream1 = new NativeArray<Stream1Data>(_totalVertexCount, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            _uvs = new NativeArray<Vector3>(_totalVertexCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            for (var i = 0; i < _drawableCount; i++)
            {
                var uvs = vertexUvs[i];
                var slice = _useTextureArray ? _textureSlot[i] : 0;
                var baseVertex = _vertexBase[i];

                for (var v = 0; v < uvs.Length; v++)
                {
                    _uvs[baseVertex + v] = new Vector3(uvs[v].x, uvs[v].y, slice);
                }

                // Initial vertex positions: the core only flags dirty drawables in its
                // dynamic data, so drawables that never move must start out correct here.
                var positions = initialPositions[i];

                if (positions != null)
                {
                    var count = Mathf.Min(positions.Length, _vertexCount[i]);

                    for (var v = 0; v < count; v++)
                    {
                        _positions[baseVertex + v] = positions[v];
                    }
                }
            }

            RebuildOrder();

            if (_controller.SortingMode == CubismSortingMode.BackToFrontZ)
            {
                RefreshSortZ(_controller.DepthOffset);
            }

            // Baked indices: per drawable local indices offset by its base vertex.
            _maskIndexCount = ComputeMaskIndexCount();
            var indexBufferCapacity = _maskIndexCount + _totalIndexCount;

            if (_use32BitIndices)
            {
                _bakedIndices32 = new NativeArray<uint>(_totalIndexCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                _indexBuffer32 = new NativeArray<uint>(indexBufferCapacity, Allocator.Persistent, NativeArrayOptions.ClearMemory);

                for (var i = 0; i < _drawableCount; i++)
                {
                    var indices = localIndices[i];
                    var baseVertex = (uint)_vertexBase[i];
                    var indexBase = _indexBase[i];
                    for (var n = 0; n < indices.Length; n++)
                    {
                        _bakedIndices32[indexBase + n] = baseVertex + (uint)indices[n];
                    }
                }
            }
            else
            {
                _bakedIndices16 = new NativeArray<ushort>(_totalIndexCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                _indexBuffer16 = new NativeArray<ushort>(indexBufferCapacity, Allocator.Persistent, NativeArrayOptions.ClearMemory);

                for (var i = 0; i < _drawableCount; i++)
                {
                    var indices = localIndices[i];
                    var baseVertex = _vertexBase[i];
                    var indexBase = _indexBase[i];
                    for (var n = 0; n < indices.Length; n++)
                    {
                        _bakedIndices16[indexBase + n] = (ushort)(baseVertex + indices[n]);
                    }
                }
            }

            // Mesh.
            _mesh = new Mesh
            {
                name = model.name + " (Batched)",
                hideFlags = HideFlags.HideAndDontSave
            };
            _mesh.MarkDynamic();

            _mesh.SetVertexBufferParams(
                _totalVertexCount,
                new VertexAttributeDescriptor(VertexAttribute.Position, VertexAttributeFormat.Float32, 3, 0),
                new VertexAttributeDescriptor(VertexAttribute.Color, VertexAttributeFormat.UNorm8, 4, 1),
                new VertexAttributeDescriptor(VertexAttribute.TexCoord0, VertexAttributeFormat.Float32, 3, 2),
                new VertexAttributeDescriptor(VertexAttribute.TexCoord1, VertexAttributeFormat.UNorm8, 4, 1),
                new VertexAttributeDescriptor(VertexAttribute.TexCoord2, VertexAttributeFormat.UNorm8, 4, 1));

            _mesh.SetIndexBufferParams(indexBufferCapacity, _use32BitIndices ? IndexFormat.UInt32 : IndexFormat.UInt16);

            // Static uploads.
            _mesh.SetVertexBufferData(_uvs, 0, 0, _totalVertexCount, 2, CubismBatchedRendering.UpdateFlags);

            // Model-wide bounds; batched draws use CommandBuffer.DrawMesh which does
            // not cull, so these just need to be sane.
            var canvas = model.CanvasInformation;
            var size = new Vector3(canvas.CanvasWidth / canvas.PixelsPerUnit, canvas.CanvasHeight / canvas.PixelsPerUnit, 1.0f);
            _mesh.bounds = new Bounds(Vector3.zero, size * 2.0f);

            // Mask atlas + static mask index region + sections.
            BuildMaskSections();

            _modelProperties = new MaterialPropertyBlock();

            _positionsDirty = true;
            _stream1Dirty = true;
            _indicesDirty = true;
        }


        private void BuildMaskGroups(CubismDrawable[] drawables)
        {
            var groupByKey = new Dictionary<string, int>();
            var members = new List<int[]>();

            for (var i = 0; i < drawables.Length; i++)
            {
                var drawable = drawables[i];
                var unmanagedIndex = drawable.UnmanagedIndex;

                if (!drawable.IsMasked || drawable.Masks.Length < 1)
                {
                    _maskGroup[unmanagedIndex] = 0;
                    continue;
                }

                var key = MaskGroupKey(drawable.Masks);

                if (!groupByKey.TryGetValue(key, out var group))
                {
                    var masks = drawable.Masks;
                    var maskIndices = new int[masks.Length];
                    for (var m = 0; m < masks.Length; m++)
                    {
                        maskIndices[m] = masks[m].UnmanagedIndex;
                    }

                    group = members.Count + 1; // Slot 0 is the "not masked" sentinel.
                    groupByKey.Add(key, group);
                    members.Add(maskIndices);
                }

                _maskGroup[unmanagedIndex] = (byte)group;
            }

            _maskGroupCount = members.Count;
            _maskGroupMembers = members.ToArray();

            _maskTiles = new Vector4[MaxMaskGroups];
            _maskTransforms = new Vector4[MaxMaskGroups];

            // Sentinel: zero channel weights and invert=1 in the vertex data produce
            // a mask factor of exactly 1 for unmasked drawables.
            _maskTiles[0] = new Vector4(-1.0f, 0.0f, 0.0f, 1.0f);
            _maskTransforms[0] = new Vector4(0.0f, 0.0f, 1.0f, 0.0f);

            if (_maskGroupCount < 1)
            {
                return;
            }

            // Tile layout: 4 channels per tile, tiles arranged in a square grid.
            var tileCount = (_maskGroupCount + 3) / 4;
            var tilesPerAxis = Mathf.CeilToInt(Mathf.Sqrt(tileCount));
            var tileSize = 1.0f / tilesPerAxis;

            for (var group = 0; group < _maskGroupCount; group++)
            {
                var channel = group & 3;
                var tileIndex = group >> 2;
                var column = tileIndex % tilesPerAxis;
                var row = tileIndex / tilesPerAxis;

                _maskTiles[group + 1] = new Vector4(channel, column, row, tileSize);
            }
        }


        private int ComputeMaskIndexCount()
        {
            var total = 0;

            for (var group = 0; group < _maskGroupCount; group++)
            {
                var groupMembers = _maskGroupMembers[group];
                for (var m = 0; m < groupMembers.Length; m++)
                {
                    total += _indexCount[groupMembers[m]];
                }
            }

            return total;
        }


        /// <summary>
        /// Builds mask atlas render texture, the static mask index region at the start
        /// of the index buffer, and the per-section materials/property blocks.
        /// Idempotent; re-run after texture changes to re-split sections.
        /// </summary>
        private void BuildMaskSections()
        {
            _maskSections.Clear();
            _maskSubMeshes.Clear();

            if (_maskGroupCount < 1)
            {
                return;
            }

            if (_maskAtlas == null)
            {
                _maskAtlas = new RenderTexture(MaskAtlasSize, MaskAtlasSize, 0, RenderTextureFormat.ARGB32)
                {
                    name = _controller.Model.name + " MaskAtlas",
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp,
                    useMipMap = false,
                    autoGenerateMips = false
                };
                _maskAtlas.Create();
            }

            var cursor = 0;
            var subMeshIndex = 0;

            for (var group = 0; group < _maskGroupCount; group++)
            {
                var groupMembers = _maskGroupMembers[group];

                // Split group members into sections by (texture, cull).
                var sectionOf = new Dictionary<long, int>();
                var sectionMembers = new List<List<int>>();
                var sectionTexture = new List<Texture>();
                var sectionDoubleSided = new List<bool>();

                for (var m = 0; m < groupMembers.Length; m++)
                {
                    var drawableIndex = groupMembers[m];
                    var renderer = _renderersByDrawable[drawableIndex];
                    var texture = renderer != null ? (Texture)renderer.MainTexture : Texture2D.whiteTexture;
                    var doubleSided = _isDoubleSided[drawableIndex];

                    var sectionKey = ((long)texture.GetInstanceID() << 1) | (doubleSided ? 1L : 0L);

                    if (!sectionOf.TryGetValue(sectionKey, out var section))
                    {
                        section = sectionMembers.Count;
                        sectionOf.Add(sectionKey, section);
                        sectionMembers.Add(new List<int>(groupMembers.Length));
                        sectionTexture.Add(texture);
                        sectionDoubleSided.Add(doubleSided);
                    }

                    sectionMembers[section].Add(drawableIndex);
                }

                for (var section = 0; section < sectionMembers.Count; section++)
                {
                    var start = cursor;

                    for (var m = 0; m < sectionMembers[section].Count; m++)
                    {
                        var drawableIndex = sectionMembers[section][m];
                        CopyBakedIndices(drawableIndex, ref cursor);
                    }

                    var properties = new MaterialPropertyBlock();
                    properties.SetTexture(MainTextureId, sectionTexture[section]);
                    properties.SetVector(MaskTileId, _maskTiles[group + 1]);

                    _maskSections.Add(new MaskSection
                    {
                        GroupIndex = group,
                        SubMeshIndex = subMeshIndex,
                        Material = sectionDoubleSided[section] ? CubismBuiltinMaterials.Mask : CubismBuiltinMaterials.MaskCulling,
                        Properties = properties
                    });

                    _maskSubMeshes.Add(MakeSubMeshDescriptor(start, cursor - start));
                    subMeshIndex++;
                }
            }

            // Upload the static mask index region.
            if (_use32BitIndices)
            {
                _mesh.SetIndexBufferData(_indexBuffer32, 0, 0, _maskIndexCount, CubismBatchedRendering.UpdateFlags);
            }
            else
            {
                _mesh.SetIndexBufferData(_indexBuffer16, 0, 0, _maskIndexCount, CubismBatchedRendering.UpdateFlags);
            }
        }


        private void CopyBakedIndices(int drawableIndex, ref int cursor)
        {
            var count = _indexCount[drawableIndex];
            var indexBase = _indexBase[drawableIndex];

            if (_use32BitIndices)
            {
                NativeArray<uint>.Copy(_bakedIndices32, indexBase, _indexBuffer32, cursor, count);
            }
            else
            {
                NativeArray<ushort>.Copy(_bakedIndices16, indexBase, _indexBuffer16, cursor, count);
            }

            cursor += count;
        }


        private SubMeshDescriptor MakeSubMeshDescriptor(int indexStart, int indexCount)
        {
            return new SubMeshDescriptor(indexStart, indexCount)
            {
                bounds = _mesh.bounds,
                firstVertex = 0,
                vertexCount = _totalVertexCount
            };
        }


        /// <summary>
        /// Builds the distinct texture table, texture slots per drawable, and the
        /// optional texture array (all textures sharing size/format/mips).
        /// </summary>
        private void BuildTextures()
        {
            var distinct = new List<Texture>(8);

            for (var i = 0; i < _drawableCount; i++)
            {
                var renderer = _renderersByDrawable[i];
                var texture = renderer != null ? (Texture)renderer.MainTexture : Texture2D.whiteTexture;

                var slot = distinct.IndexOf(texture);
                if (slot < 0)
                {
                    slot = distinct.Count;
                    distinct.Add(texture);
                }

                _textureSlot[i] = slot;
            }

            _textures = distinct.ToArray();
            _useTextureArray = false;

            if (CubismBatchedRendering.TextureArrayAllowed && _textures.Length > 1)
            {
                TryBuildTextureArray();
            }
        }


        private void TryBuildTextureArray()
        {
            var first = _textures[0] as Texture2D;

            if (first == null)
            {
                return;
            }

            for (var i = 1; i < _textures.Length; i++)
            {
                var texture = _textures[i] as Texture2D;

                if (texture == null
                    || texture.width != first.width
                    || texture.height != first.height
                    || texture.graphicsFormat != first.graphicsFormat
                    || texture.mipmapCount != first.mipmapCount)
                {
                    return;
                }
            }

            try
            {
                var array = new Texture2DArray(
                    first.width, first.height, _textures.Length,
                    first.graphicsFormat,
                    first.mipmapCount > 1 ? UnityEngine.Experimental.Rendering.TextureCreationFlags.MipChain : UnityEngine.Experimental.Rendering.TextureCreationFlags.None,
                    first.mipmapCount)
                {
                    name = _controller.Model.name + " TextureArray",
                    filterMode = first.filterMode,
                    wrapMode = first.wrapMode,
                    anisoLevel = first.anisoLevel
                };

                if (array.mipmapCount != first.mipmapCount)
                {
                    UnityEngine.Object.DestroyImmediate(array);
                    return;
                }

                for (var i = 0; i < _textures.Length; i++)
                {
                    Graphics.CopyTexture(_textures[i], 0, array, i);
                }

                _textureArray = array;
                _useTextureArray = true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[CubismBatchedModelRenderer] Texture array unavailable, batching per texture instead: {e.Message}");

                if (_textureArray != null)
                {
                    UnityEngine.Object.DestroyImmediate(_textureArray);
                    _textureArray = null;
                }

                _useTextureArray = false;
            }
        }


        private Material GetBatchMaterial(int textureSlot, BlendTypes.ColorBlend colorBlend, bool isDoubleSided)
        {
            var key = ((long)(_useTextureArray ? 0 : textureSlot) << 8)
                      | ((long)colorBlend << 2)
                      | (isDoubleSided ? 0L : 2L)
                      | (_useTextureArray ? 1L : 0L);

            if (_materials.TryGetValue(key, out var material) && material != null)
            {
                return material;
            }

            material = new Material(CubismBatchedRendering.Shader)
            {
                name = $"Cubism Batched ({colorBlend}{(isDoubleSided ? string.Empty : ", Cull")}{(_useTextureArray ? ", Array" : $", Tex{textureSlot}")})",
                hideFlags = HideFlags.HideAndDontSave
            };

            switch (colorBlend)
            {
                case BlendTypes.ColorBlend.Add:
                    material.SetInt(SrcColorId, (int)UnityEngine.Rendering.BlendMode.One);
                    material.SetInt(DstColorId, (int)UnityEngine.Rendering.BlendMode.One);
                    material.SetInt(SrcAlphaId, (int)UnityEngine.Rendering.BlendMode.Zero);
                    material.SetInt(DstAlphaId, (int)UnityEngine.Rendering.BlendMode.One);
                    break;
                case BlendTypes.ColorBlend.Multiply:
                    material.SetInt(SrcColorId, (int)UnityEngine.Rendering.BlendMode.DstColor);
                    material.SetInt(DstColorId, (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                    material.SetInt(SrcAlphaId, (int)UnityEngine.Rendering.BlendMode.Zero);
                    material.SetInt(DstAlphaId, (int)UnityEngine.Rendering.BlendMode.One);
                    break;
                default:
                    material.SetInt(SrcColorId, (int)UnityEngine.Rendering.BlendMode.One);
                    material.SetInt(DstColorId, (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                    material.SetInt(SrcAlphaId, (int)UnityEngine.Rendering.BlendMode.One);
                    material.SetInt(DstAlphaId, (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                    break;
            }

            material.SetInt(CullId, isDoubleSided ? (int)CullMode.Off : (int)CullMode.Back);

            if (_useTextureArray)
            {
                material.EnableKeyword("CUBISM_TEXTURE_ARRAY");
                material.SetTexture(MainTextureArrayId, _textureArray);
            }
            else
            {
                material.SetTexture(MainTextureId, _textures[textureSlot]);
            }

            _materials[key] = material;

            return material;
        }

        #endregion


        #region Per-Frame Update (main thread, from OnDynamicDrawableData)

        /// <summary>
        /// Consumes new dynamic data from the core. Runs on the main thread inside
        /// <see cref="CubismModel.OnDynamicDrawableData"/>, i.e. while the core task
        /// is guaranteed idle.
        /// </summary>
        public unsafe void ConsumeDynamicData(CubismDynamicDrawableData[] data)
        {
            if (!IsValid || data == null || data.Length != _drawableCount)
            {
                return;
            }

            var fullRefresh = !_receivedFirstData;
            _receivedFirstData = true;

            var orderDirty = false;
            var visibilityDirty = false;
            var applySortZ = _controller.SortingMode == CubismSortingMode.BackToFrontZ;
            var depthOffset = _controller.DepthOffset;

            var positions = (Vector3*)_positions.GetUnsafePtr();

            for (var i = 0; i < _drawableCount; i++)
            {
                var drawableData = data[i];

                // Positions must only be pulled when flagged dirty: the core skips
                // copying vertex data of non-dirty drawables into the dynamic buffers,
                // so those arrays stay zero. Initial values come from Build().
                if (drawableData.AreVertexPositionsDirty)
                {
                    var source = drawableData.VertexPositions;
                    var baseVertex = _vertexBase[i];
                    var count = _vertexCount[i];

                    if (source != null && source.Length >= count)
                    {
                        fixed (Vector3* sourcePointer = source)
                        {
                            UnsafeUtility.MemCpy(positions + baseVertex, sourcePointer, (long)count * sizeof(Vector3));
                        }

                        if (applySortZ)
                        {
                            var z = _renderOrders[i] * -depthOffset;
                            for (var v = 0; v < count; v++)
                            {
                                positions[baseVertex + v].z = z;
                            }
                        }
                    }

                    _positionsDirty = true;
                }

                if (fullRefresh || drawableData.IsRenderOrderDirty)
                {
                    if (_renderOrders[i] != drawableData.RenderOrder)
                    {
                        _renderOrders[i] = drawableData.RenderOrder;
                        orderDirty = true;
                    }
                }

                if (fullRefresh || drawableData.IsVisibilityDirty)
                {
                    var isVisible = drawableData.IsVisible;

                    if (_visible[i] != isVisible)
                    {
                        _visible[i] = isVisible;
                        visibilityDirty = true;
                    }

                    // Keep the (mesh-less) MeshRenderer's enabled flag in sync;
                    // raycasting and user code use it as the visibility signal.
                    var renderer = _renderersByDrawable[i];
                    if (renderer != null && renderer.MeshRenderer.enabled != isVisible)
                    {
                        renderer.MeshRenderer.enabled = isVisible;
                    }
                }

                if (fullRefresh || drawableData.IsOpacityDirty)
                {
                    _opacities[i] = drawableData.Opacity;

                    var renderer = _renderersByDrawable[i];
                    if (renderer != null)
                    {
                        renderer.Opacity = drawableData.Opacity;
                    }

                    RecomputeColorRow(i);
                }
                else if (fullRefresh || drawableData.IsBlendColorDirty)
                {
                    RecomputeColorRow(i);
                }
            }

            if (orderDirty)
            {
                RebuildOrder();
            }

            if (orderDirty || visibilityDirty)
            {
                _indicesDirty = true;
            }

            if (orderDirty && applySortZ)
            {
                // Depth offsets follow render order; refresh z on next position pass.
                RefreshSortZ(depthOffset);
            }
        }


        private unsafe void RefreshSortZ(float depthOffset)
        {
            var positions = (Vector3*)_positions.GetUnsafePtr();

            for (var i = 0; i < _drawableCount; i++)
            {
                var z = _renderOrders[i] * -depthOffset;
                var baseVertex = _vertexBase[i];
                var count = _vertexCount[i];

                for (var v = 0; v < count; v++)
                {
                    positions[baseVertex + v].z = z;
                }
            }

            _positionsDirty = true;
        }


        private void RebuildOrder()
        {
            // Render orders form a permutation of [0, drawableCount).
            var isPermutation = true;

            for (var i = 0; i < _drawableCount; i++)
            {
                _orderedDrawables[i] = -1;
            }

            for (var i = 0; i < _drawableCount; i++)
            {
                var order = _renderOrders[i];

                if (order < 0 || order >= _drawableCount || _orderedDrawables[order] != -1)
                {
                    isPermutation = false;
                    break;
                }

                _orderedDrawables[order] = i;
            }

            if (isPermutation)
            {
                return;
            }

            // Defensive fallback: stable sort by render order.
            for (var i = 0; i < _drawableCount; i++)
            {
                _orderedDrawables[i] = i;
            }

            var orders = _renderOrders;
            Array.Sort(_orderedDrawables, (a, b) =>
            {
                var byOrder = orders[a].CompareTo(orders[b]);
                return byOrder != 0 ? byOrder : a.CompareTo(b);
            });
        }


        /// <summary>
        /// Recomputes the stream-1 row (tint/opacity, multiply, screen, mask attrs) of one drawable.
        /// </summary>
        public void RecomputeColorRow(int drawableIndex)
        {
            if (_isDisposed || _isBroken)
            {
                return;
            }

            var renderer = _renderersByDrawable[drawableIndex];

            if (renderer == null)
            {
                return;
            }

            var tint = renderer.Color;
            tint.a *= _opacities[drawableIndex];

            var multiply = renderer.MultiplyColor;
            var screen = renderer.ScreenColor;

            if (_convertBlendColorsToLinear)
            {
                multiply = multiply.linear;
                screen = screen.linear;
            }

            var row = new Stream1Data
            {
                Color = tint,
                MultiplyAndGroup = new Color32(
                    (byte)Mathf.Clamp(Mathf.RoundToInt(multiply.r * 255.0f), 0, 255),
                    (byte)Mathf.Clamp(Mathf.RoundToInt(multiply.g * 255.0f), 0, 255),
                    (byte)Mathf.Clamp(Mathf.RoundToInt(multiply.b * 255.0f), 0, 255),
                    _maskGroup[drawableIndex]),
                ScreenAndInvert = new Color32(
                    (byte)Mathf.Clamp(Mathf.RoundToInt(screen.r * 255.0f), 0, 255),
                    (byte)Mathf.Clamp(Mathf.RoundToInt(screen.g * 255.0f), 0, 255),
                    (byte)Mathf.Clamp(Mathf.RoundToInt(screen.b * 255.0f), 0, 255),
                    // Unmasked sentinel needs invert=1 so the mask factor is 1.
                    (byte)(_maskGroup[drawableIndex] == 0 ? 255 : (_isInverted[drawableIndex] ? 255 : 0)))
            };

            var baseVertex = _vertexBase[drawableIndex];
            var count = _vertexCount[drawableIndex];

            for (var v = 0; v < count; v++)
            {
                _stream1[baseVertex + v] = row;
            }

            _stream1Dirty = true;
        }


        /// <summary>
        /// Recomputes the stream-1 row for a renderer (hook target for color changes).
        /// </summary>
        public void MarkColorDirty(CubismRenderer renderer)
        {
            if (renderer == null || renderer.Drawable == null)
            {
                return;
            }

            RecomputeColorRow(renderer.Drawable.UnmanagedIndex);
        }


        /// <summary>
        /// Requests a texture table + material + batch rebuild (hook target for texture changes).
        /// </summary>
        public void MarkTexturesDirty()
        {
            _texturesDirty = true;
        }

        #endregion


        #region Rendering (main thread, from the URP render pass)

        /// <summary>
        /// Uploads dirty CPU buffers into the mesh. Called once per frame before recording draws.
        /// </summary>
        public void FlushMeshData()
        {
            if (!IsValid)
            {
                return;
            }

            if (_lastFlushedFrame == Time.frameCount && !_texturesDirty)
            {
                return;
            }

            _lastFlushedFrame = Time.frameCount;

            if (_texturesDirty)
            {
                RebuildTexturesAndUvs();
                _texturesDirty = false;
                _indicesDirty = true;
            }

            if (_positionsDirty)
            {
                _mesh.SetVertexBufferData(_positions, 0, 0, _totalVertexCount, 0, CubismBatchedRendering.UpdateFlags);
                _positionsDirty = false;
            }

            if (_stream1Dirty)
            {
                _mesh.SetVertexBufferData(_stream1, 0, 0, _totalVertexCount, 1, CubismBatchedRendering.UpdateFlags);
                _stream1Dirty = false;
            }

            if (_indicesDirty)
            {
                RebuildBatches();
                _indicesDirty = false;
            }
        }


        private void RebuildTexturesAndUvs()
        {
            // Release materials bound to the old texture set.
            foreach (var material in _materials.Values)
            {
                if (material != null)
                {
                    UnityEngine.Object.DestroyImmediate(material);
                }
            }
            _materials.Clear();

            if (_textureArray != null)
            {
                UnityEngine.Object.DestroyImmediate(_textureArray);
                _textureArray = null;
            }

            BuildTextures();

            // Refresh texture slices in the static uv stream.
            for (var i = 0; i < _drawableCount; i++)
            {
                var slice = _useTextureArray ? _textureSlot[i] : 0;
                var baseVertex = _vertexBase[i];
                var count = _vertexCount[i];

                for (var v = 0; v < count; v++)
                {
                    var uv = _uvs[baseVertex + v];
                    uv.z = slice;
                    _uvs[baseVertex + v] = uv;
                }
            }

            _mesh.SetVertexBufferData(_uvs, 0, 0, _totalVertexCount, 2, CubismBatchedRendering.UpdateFlags);

            // Mask sections split by texture, so their layout may have changed too.
            BuildMaskSections();
        }


        private void RebuildBatches()
        {
            _batches.Clear();

            var cursor = _maskIndexCount;
            var batchStart = cursor;
            var hasOpenBatch = false;
            var currentTexture = -1;
            var currentBlend = BlendTypes.ColorBlend.Normal;
            var currentDoubleSided = false;

            for (var order = 0; order < _drawableCount; order++)
            {
                var drawableIndex = _orderedDrawables[order];

                if (drawableIndex < 0
                    || !_visible[drawableIndex]
                    || _indexCount[drawableIndex] < 1)
                {
                    continue;
                }

                var textureSlot = _useTextureArray ? 0 : _textureSlot[drawableIndex];
                var blend = _colorBlend[drawableIndex];
                var doubleSided = _isDoubleSided[drawableIndex];

                if (!hasOpenBatch
                    || textureSlot != currentTexture
                    || blend != currentBlend
                    || doubleSided != currentDoubleSided)
                {
                    if (hasOpenBatch)
                    {
                        _batches.Add(new Batch
                        {
                            TextureSlot = currentTexture,
                            ColorBlend = currentBlend,
                            IsDoubleSided = currentDoubleSided,
                            IndexStart = batchStart,
                            IndexCount = cursor - batchStart
                        });
                    }

                    batchStart = cursor;
                    currentTexture = textureSlot;
                    currentBlend = blend;
                    currentDoubleSided = doubleSided;
                    hasOpenBatch = true;
                }

                CopyBakedIndices(drawableIndex, ref cursor);
            }

            if (hasOpenBatch && cursor > batchStart)
            {
                _batches.Add(new Batch
                {
                    TextureSlot = currentTexture,
                    ColorBlend = currentBlend,
                    IsDoubleSided = currentDoubleSided,
                    IndexStart = batchStart,
                    IndexCount = cursor - batchStart
                });
            }

            _mainIndexCount = cursor - _maskIndexCount;

            // Upload the rebuilt main region.
            if (_mainIndexCount > 0)
            {
                if (_use32BitIndices)
                {
                    _mesh.SetIndexBufferData(_indexBuffer32, _maskIndexCount, _maskIndexCount, _mainIndexCount, CubismBatchedRendering.UpdateFlags);
                }
                else
                {
                    _mesh.SetIndexBufferData(_indexBuffer16, _maskIndexCount, _maskIndexCount, _mainIndexCount, CubismBatchedRendering.UpdateFlags);
                }
            }

            // Apply the whole submesh table (mask sections first, then batches) in a
            // single call; growing subMeshCount incrementally spams Unity's invalid
            // AABB conversion warning for the transiently-default descriptors.
            _subMeshScratch.Clear();
            _subMeshScratch.AddRange(_maskSubMeshes);

            for (var batch = 0; batch < _batches.Count; batch++)
            {
                _subMeshScratch.Add(MakeSubMeshDescriptor(_batches[batch].IndexStart, _batches[batch].IndexCount));
            }

            _mesh.SetSubMeshes(_subMeshScratch, CubismBatchedRendering.UpdateFlags);
        }


        /// <summary>
        /// Records the mask atlas pass. Call before the main render target is set.
        /// </summary>
        public void RecordMaskPass(CommandBuffer buffer)
        {
            if (!IsValid || _maskSections.Count < 1)
            {
                return;
            }

            UpdateMaskTransforms();

            buffer.SetRenderTarget(_maskAtlas);
            buffer.ClearRenderTarget(false, true, Color.clear);

            for (var section = 0; section < _maskSections.Count; section++)
            {
                var maskSection = _maskSections[section];

                maskSection.Properties.SetVector(MaskTransformId, _maskTransforms[maskSection.GroupIndex + 1]);

                buffer.DrawMesh(
                    _mesh,
                    Matrix4x4.identity,
                    maskSection.Material,
                    maskSection.SubMeshIndex,
                    0,
                    maskSection.Properties);
            }
        }


        private unsafe void UpdateMaskTransforms()
        {
            if (_lastMaskUpdateFrame == Time.frameCount)
            {
                return;
            }

            _lastMaskUpdateFrame = Time.frameCount;

            var positions = (Vector3*)_positions.GetUnsafePtr();

            for (var group = 0; group < _maskGroupCount; group++)
            {
                var groupMembers = _maskGroupMembers[group];

                var min = new Vector2(float.MaxValue, float.MaxValue);
                var max = new Vector2(float.MinValue, float.MinValue);

                for (var m = 0; m < groupMembers.Length; m++)
                {
                    var drawableIndex = groupMembers[m];
                    var baseVertex = _vertexBase[drawableIndex];
                    var count = _vertexCount[drawableIndex];

                    for (var v = 0; v < count; v++)
                    {
                        var position = positions[baseVertex + v];

                        if (position.x < min.x) { min.x = position.x; }
                        if (position.y < min.y) { min.y = position.y; }
                        if (position.x > max.x) { max.x = position.x; }
                        if (position.y > max.y) { max.y = position.y; }
                    }
                }

                var size = max - min;
                var scale = Mathf.Max(size.x, size.y);

                if (scale < 1e-6f || min.x > max.x)
                {
                    scale = 1e-6f;
                }

                var center = (min + max) * 0.5f;

                _maskTransforms[group + 1] = new Vector4(center.x, center.y, 1.0f / scale, 0.0f);
            }
        }


        /// <summary>
        /// Records main draws for all batches. The caller must have set the render target.
        /// </summary>
        public void RecordMainDraws(CommandBuffer buffer)
        {
            if (!IsValid || _batches.Count < 1)
            {
                return;
            }

            var controllerTransform = _controller.transform;
            var matrix = Matrix4x4.TRS(
                controllerTransform.localPosition,
                controllerTransform.localRotation,
                controllerTransform.localScale);

            _modelProperties.SetFloat(ModelOpacityId, Mathf.Clamp01(_controller.Opacity));
            _modelProperties.SetVectorArray(MaskTilesArrayId, _maskTiles);
            _modelProperties.SetVectorArray(MaskTransformsArrayId, _maskTransforms);
            _modelProperties.SetTexture(MaskTextureId, _maskAtlas != null ? (Texture)_maskAtlas : Texture2D.whiteTexture);

            for (var batch = 0; batch < _batches.Count; batch++)
            {
                var currentBatch = _batches[batch];
                var material = GetBatchMaterial(currentBatch.TextureSlot, currentBatch.ColorBlend, currentBatch.IsDoubleSided);

                buffer.DrawMesh(
                    _mesh,
                    matrix,
                    material,
                    _maskSections.Count + batch,
                    0,
                    _modelProperties);
            }
        }

        #endregion


        #region Disposal

        /// <summary>
        /// Refreshes all dynamic state from the model after the controller was
        /// disabled and re-enabled (e.g. avatar power management toggling the
        /// render controller on screen changes). Keeps the expensive GPU/native
        /// resources alive so the resume is stutter-free; only CPU-side state and
        /// the next frame's uploads are refreshed. The core does not run while the
        /// controller is disabled in the supported flows, but dirty flags emitted
        /// during the gap are lost, so everything is re-read defensively.
        /// </summary>
        /// <returns>False when the renderer no longer matches the model and must be rebuilt.</returns>
        public bool ResumeAfterDisable()
        {
            if (!IsValid)
            {
                return false;
            }

            var model = _controller.Model;

            if (model == null
                || model.Drawables == null
                || model.Drawables.Length != _drawableCount)
            {
                return false;
            }

            var drawables = model.Drawables;
            var renderOrders = model.AllDrawObjectsRenderOrder;

            for (var i = 0; i < drawables.Length; i++)
            {
                var drawable = drawables[i];
                var unmanagedIndex = drawable.UnmanagedIndex;

                if (unmanagedIndex < 0 || unmanagedIndex >= _drawableCount)
                {
                    return false;
                }

                // Current pose (the core may have been updated while unsubscribed).
                var positions = drawable.VertexPositions;
                var baseVertex = _vertexBase[unmanagedIndex];
                var count = Mathf.Min(_vertexCount[unmanagedIndex], positions != null ? positions.Length : 0);

                for (var v = 0; v < count; v++)
                {
                    var position = _positions[baseVertex + v];
                    position.x = positions[v].x;
                    position.y = positions[v].y;
                    _positions[baseVertex + v] = position;
                }

                _renderOrders[unmanagedIndex] = renderOrders[unmanagedIndex];

                RecomputeColorRow(unmanagedIndex);
            }

            RebuildOrder();

            if (_controller.SortingMode == CubismSortingMode.BackToFrontZ)
            {
                RefreshSortZ(_controller.DepthOffset);
            }

            _positionsDirty = true;
            _stream1Dirty = true;
            _indicesDirty = true;
            _lastFlushedFrame = -1;
            _lastMaskUpdateFrame = -1;

            return true;
        }


        /// <summary>
        /// Pushes the batched path's dynamic state (visibility, render orders) back
        /// onto the per-drawable renderers. Call before falling back to the legacy
        /// path at runtime; the legacy event flow only propagates dirty changes, so
        /// state that changed while batched would otherwise stay stale.
        /// </summary>
        public void RestoreLegacyRendererState()
        {
            if (_isBroken || _renderersByDrawable == null)
            {
                return;
            }

            // Skip during scene teardown; the renderers are being destroyed anyway.
            if (_controller == null || !_controller.gameObject.scene.isLoaded)
            {
                return;
            }

            for (var i = 0; i < _drawableCount; i++)
            {
                var renderer = _renderersByDrawable[i];

                if (renderer == null)
                {
                    continue;
                }

                renderer.MeshRenderer.enabled = _visible[i];
                renderer.SetDrawObjectRenderOrder(_renderOrders[i]);
            }
        }


        /// <summary>
        /// Releases all GPU and native resources.
        /// </summary>
        public void Dispose()
        {
            if (_isDisposed)
            {
                return;
            }

            _isDisposed = true;
            DisposeResources();
        }


        private void DisposeResources()
        {
            if (_positions.IsCreated) { _positions.Dispose(); }
            if (_stream1.IsCreated) { _stream1.Dispose(); }
            if (_uvs.IsCreated) { _uvs.Dispose(); }
            if (_bakedIndices16.IsCreated) { _bakedIndices16.Dispose(); }
            if (_bakedIndices32.IsCreated) { _bakedIndices32.Dispose(); }
            if (_indexBuffer16.IsCreated) { _indexBuffer16.Dispose(); }
            if (_indexBuffer32.IsCreated) { _indexBuffer32.Dispose(); }

            if (_mesh != null)
            {
                UnityEngine.Object.DestroyImmediate(_mesh);
                _mesh = null;
            }

            if (_maskAtlas != null)
            {
                _maskAtlas.Release();
                UnityEngine.Object.DestroyImmediate(_maskAtlas);
                _maskAtlas = null;
            }

            if (_textureArray != null)
            {
                UnityEngine.Object.DestroyImmediate(_textureArray);
                _textureArray = null;
            }

            foreach (var material in _materials.Values)
            {
                if (material != null)
                {
                    UnityEngine.Object.DestroyImmediate(material);
                }
            }
            _materials.Clear();
        }

        #endregion
    }
}
