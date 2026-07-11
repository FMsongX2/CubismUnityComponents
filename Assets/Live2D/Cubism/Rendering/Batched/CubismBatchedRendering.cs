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
        /// clears, and the final blit. Additive/multiplicative drawables then blend
        /// against the scene behind the model (identical to the pre-5.2 renderer)
        /// instead of against a transparent buffer.
        /// </summary>
        public static bool DrawToCameraTargetDirectly = true;

        /// <summary>
        /// When true, models whose textures share size/format/mips get a runtime
        /// <see cref="Texture2DArray"/> so texture switches stop splitting batches.
        /// </summary>
        public static bool UseTextureArray = true;

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
