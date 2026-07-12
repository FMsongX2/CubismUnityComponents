/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */


using UnityEngine;
using UnityEngine.Rendering;


namespace Live2D.Cubism.Rendering
{
    /// <summary>
    /// Global switches and shared resources for the mobile batched fast path.
    /// </summary>
    public static class CubismBatchedRendering
    {
        /// <summary>
        /// Master switch. When false every model renders through the legacy path.
        /// Changes take effect for controllers enabled afterwards.
        /// </summary>
        public static bool Enabled = true;

        /// <summary>
        /// When true and every render controller group is batched, models draw straight
        /// into the camera target, skipping the intermediate full-screen texture, its
        /// clears, and the final blit (~0.4ms GPU on a 2024 flagship). Additive and
        /// multiplicative drawables then blend against the scene behind the model
        /// (identical to the pre-5.2 renderer) instead of against a transparent buffer.
        /// Off by default: the buffered composition keeps the exact legacy blend
        /// semantics for any model content and renders through the same offscreen
        /// texture + blit sequence the stock pipeline has always used.
        /// </summary>
        public static bool DrawToCameraTargetDirectly = false;

        /// <summary>
        /// When true, models whose textures share size/format/mips get a runtime
        /// <see cref="Texture2DArray"/> so texture switches stop splitting batches.
        /// The array duplicates the source textures in memory, so it is skipped on
        /// devices below <see cref="TextureArrayMinimumSystemMemoryMegabytes"/>;
        /// per-texture batching still applies there.
        /// </summary>
        public static bool UseTextureArray = true;

        /// <summary>
        /// Minimum <see cref="SystemInfo.systemMemorySize"/> (MB) for the runtime
        /// texture array. Below this, batching splits per texture instead of
        /// spending an extra texture-set worth of memory. 0 disables the check.
        /// </summary>
        public static int TextureArrayMinimumSystemMemoryMegabytes = 4096;

        /// <summary>
        /// Seconds to wait after a model (re)build before snapshotting the source
        /// textures into the runtime <see cref="Texture2DArray"/>. Textures still in
        /// the async GPU upload queue (scene load, app cold start) can hold
        /// placeholder content; a per-texture binding follows the upload
        /// transparently, but the array copy would freeze that placeholder
        /// permanently. Until the window passes the model batches per texture
        /// (visually identical). 0 copies immediately.
        /// </summary>
        public static float TextureArrayActivationDelaySeconds = 3.0f;

        /// <summary>
        /// Effective texture-array switch after device constraints.
        /// </summary>
        internal static bool TextureArrayAllowed
        {
            get
            {
                return UseTextureArray
                       && (TextureArrayMinimumSystemMemoryMegabytes <= 0
                           || SystemInfo.systemMemorySize >= TextureArrayMinimumSystemMemoryMegabytes);
            }
        }

        /// <summary>
        /// Mesh update flags used for all batched mesh uploads.
        /// </summary>
        internal const MeshUpdateFlags UpdateFlags =
            MeshUpdateFlags.DontValidateIndices
            | MeshUpdateFlags.DontNotifyMeshUsers
            | MeshUpdateFlags.DontRecalculateBounds
            | MeshUpdateFlags.DontResetBoneBounds;


        /// <summary>
        /// <see cref="Shader"/> backing field.
        /// </summary>
        private static Shader _shader;

        /// <summary>
        /// The batched drawable shader.
        /// </summary>
        public static Shader Shader
        {
            get
            {
                if (_shader == null)
                {
                    _shader = Resources.Load<Shader>("Live2D/Cubism/Shaders/BlendMode/UnlitBatched");
                }

                if (_shader == null)
                {
                    _shader = UnityEngine.Shader.Find("Live2D Cubism/Batched");
                }

                return _shader;
            }
        }
    }
}
