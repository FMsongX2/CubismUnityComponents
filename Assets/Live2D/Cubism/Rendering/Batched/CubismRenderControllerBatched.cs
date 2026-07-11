/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */


using Live2D.Cubism.Core;
using Live2D.Cubism.Rendering.URP.RenderingInterceptor;
using UnityEngine;


namespace Live2D.Cubism.Rendering
{
    /// <summary>
    /// Mobile batched fast path state of the render controller.
    /// </summary>
    public sealed partial class CubismRenderController
    {
        /// <summary>
        /// Set to force this model through the legacy per-drawable pipeline even
        /// when it qualifies for batched rendering.
        /// </summary>
        [SerializeField, HideInInspector]
        public bool ForceLegacyRendering;

        /// <summary>
        /// True while this model renders through <see cref="CubismBatchedModelRenderer"/>.
        /// </summary>
        public bool IsBatchedRenderingActive { get; private set; }

        /// <summary>
        /// Batched renderer instance (null unless active).
        /// </summary>
        internal CubismBatchedModelRenderer BatchedRenderer { get; private set; }


        /// <summary>
        /// Decides whether the batched fast path applies to this model. Must run
        /// before renderers initialize so legacy per-drawable meshes can be skipped.
        /// </summary>
        private void TryActivateBatchedRendering()
        {
            IsBatchedRenderingActive = false;

            if (!Application.isPlaying
                || !CubismBatchedRendering.Enabled
                || ForceLegacyRendering
                || CubismBatchedRendering.Shader == null
                || !Model)
            {
                return;
            }

            // Rendering interceptors need per-drawable draw events.
            if (GetComponent<ICubismRenderingInterceptor>() != null
                || CubismRenderingInterceptorsManager.GetInstance().Interceptors.Length > 0)
            {
                return;
            }

            if (!CubismBatchedModelRenderer.IsModelEligible(this))
            {
                return;
            }

            IsBatchedRenderingActive = true;
        }


        /// <summary>
        /// Creates the batched renderer once renderers are initialized.
        /// </summary>
        private void TryInitializeBatchedRenderer()
        {
            if (!IsBatchedRenderingActive || BatchedRenderer != null)
            {
                return;
            }

            if (CubismBatchedModelRenderer.AreRenderersEligible(this))
            {
                BatchedRenderer = new CubismBatchedModelRenderer(this);
            }

            if (BatchedRenderer == null || !BatchedRenderer.IsValid)
            {
                // Initialization failed; renderers already skipped their meshes, so
                // rebuild them for the legacy path.
                BatchedRenderer?.Dispose();
                BatchedRenderer = null;
                IsBatchedRenderingActive = false;

                var renderers = Renderers;
                for (var i = 0; i < renderers.Length; i++)
                {
                    renderers[i].TryInitialize(this);
                }
            }
        }


        /// <summary>
        /// Releases the batched renderer.
        /// </summary>
        private void DisposeBatchedRenderer()
        {
            if (BatchedRenderer != null)
            {
                // Keep per-drawable renderers consistent in case the model comes back
                // on the legacy path.
                if (Application.isPlaying)
                {
                    BatchedRenderer.RestoreLegacyRendererState();
                }

                BatchedRenderer.Dispose();
                BatchedRenderer = null;
            }

            IsBatchedRenderingActive = false;
        }


        /// <summary>
        /// Fast-path consumption of new dynamic core data.
        /// </summary>
        /// <returns>True when handled (legacy per-renderer processing must be skipped).</returns>
        private bool TryConsumeDynamicDataBatched(CubismModel sender, CubismDynamicDrawableData[] data)
        {
            if (!IsBatchedRenderingActive)
            {
                return false;
            }

            TryInitializeBatchedRenderer();

            if (!IsBatchedRenderingActive || BatchedRenderer == null)
            {
                return false;
            }

            BatchedRenderer.ConsumeDynamicData(data);

            // Preserve public handler callbacks.
            var drawOrderHandler = DrawOrderHandlerInterface;

            if (drawOrderHandler != null)
            {
                var drawables = sender.Drawables;

                for (var i = 0; i < data.Length; ++i)
                {
                    if (data[i].IsDrawOrderDirty)
                    {
                        drawOrderHandler.OnDrawOrderDidChange(this, drawables[i], data[i].DrawOrder);
                    }
                }
            }

            return true;
        }
    }
}
