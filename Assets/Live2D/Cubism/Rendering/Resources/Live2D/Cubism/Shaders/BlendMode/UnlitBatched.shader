/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */


// Batched drawable shader for the mobile fast path.
// All per-drawable state (multiply/screen color, mask tile, invert flag) is carried
// in vertex attributes so consecutive drawables sharing (texture, blend, cull)
// render as a single draw call. Mask parameters are looked up per vertex from
// property-block arrays filled once per model per frame.
Shader "Live2D Cubism/Batched"
{
    Properties
    {
        [PerRendererData] _MainTex ("Texture", 2D) = "white" {}
        [PerRendererData] cubism_ModelOpacity ("Model Opacity", Float) = 1

        [PerRendererData] cubism_MaskTexture ("cubism_Internal", 2D) = "white" {}

        // Blend settings.
        _SrcColor ("Source Color", Int) = 1
        _DstColor ("Destination Color", Int) = 10
        _SrcAlpha ("Source Alpha", Int) = 1
        _DstAlpha ("Destination Alpha", Int) = 10

        // Culling setting.
        _Cull ("Culling", Int) = 0
    }
    SubShader
    {
        Blend [_SrcColor][_DstColor], [_SrcAlpha][_DstAlpha]

        Tags {
            "Queue" = "Transparent"
            "IgnoreProjector" = "True"
            "RenderType" = "Transparent"
            "PreviewType" = "Plane"
            "CanUseSpriteAtlas" = "True"
            "RenderPipeline" = "UniversalPipeline"
        }

        Cull     [_Cull]
        Lighting Off
        ZWrite   Off
        ZTest    LEqual

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_local _ CUBISM_TEXTURE_ARRAY

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "../CubismCG.cginc"

            struct Attributes
            {
                float3 vertex   : POSITION;
                float4 color    : COLOR;      // tint.rgb, tint.a * drawable opacity
                float3 texcoord : TEXCOORD0;  // uv.xy, texture array slice
                float4 channel1 : TEXCOORD1;  // multiply.rgb, mask group index / 255
                float4 channel2 : TEXCOORD2;  // screen.rgb, invert flag
            };

            struct Varyings
            {
                float4 vertex           : SV_POSITION;
                float4 color            : COLOR;
                float3 texcoord         : TEXCOORD0;
                float3 multiplyColor    : TEXCOORD1;
                float4 screenColorInvert: TEXCOORD2;
                float2 maskCoord        : TEXCOORD3;
                float4 maskTile         : TEXCOORD4;
            };

            // Per-model mask parameters, indexed by the vertex mask group.
            // Group slot with tile.x < 0 means "not masked" (channel weights all zero;
            // paired with invert = 1 the mask factor becomes exactly 1).
            float4 _CubismMaskTiles[64];
            float4 _CubismMaskTransforms[64];

            float cubism_ModelOpacity;

            sampler2D cubism_MaskTexture;

#if defined(CUBISM_TEXTURE_ARRAY)
            TEXTURE2D_ARRAY(_MainTexArray);
            SAMPLER(sampler_MainTexArray);
#else
            sampler2D _MainTex;
#endif

            Varyings vert (Attributes IN)
            {
                Varyings OUT;

                OUT.vertex = TransformObjectToHClip(IN.vertex);
                OUT.color = IN.color;
                OUT.texcoord = IN.texcoord;
                OUT.multiplyColor = IN.channel1.rgb;
                OUT.screenColorInvert = IN.channel2;

                // Look up this drawable's mask group parameters.
                int maskGroup = (int)round(IN.channel1.a * 255.0);
                float4 tile = _CubismMaskTiles[maskGroup];
                float4 maskTransform = _CubismMaskTransforms[maskGroup];

                OUT.maskCoord = CubismToMaskCoordinates(IN.vertex.xy, tile, maskTransform);
                OUT.maskTile = tile;

                return OUT;
            }

            half4 frag (Varyings IN) : SV_Target
            {
#if defined(CUBISM_TEXTURE_ARRAY)
                half4 textureColor = SAMPLE_TEXTURE2D_ARRAY(_MainTexArray, sampler_MainTexArray, IN.texcoord.xy, IN.texcoord.z);
#else
                half4 textureColor = tex2D(_MainTex, IN.texcoord.xy);
#endif

                // Multiply color.
                textureColor.rgb *= IN.multiplyColor;
                // Screen color.
                textureColor.rgb = (textureColor.rgb + IN.screenColorInvert.rgb) - (textureColor.rgb * IN.screenColorInvert.rgb);

                half4 OUT = textureColor * IN.color;

                // Premultiply alpha.
                OUT.rgb *= OUT.a;

                // Clipping mask (atlas channel select; factor is 1 for unmasked groups).
                float4 maskChannel = CubismGetClippedMaskChannel(IN.maskCoord, IN.maskTile);
                float maskAlpha = CubismSampleMaskTexture(cubism_MaskTexture, maskChannel, IN.maskCoord);
                OUT *= lerp(maskAlpha, 1.0 - maskAlpha, IN.screenColorInvert.a);

                OUT *= cubism_ModelOpacity;

                return OUT;
            }
            ENDHLSL
        }
    }
}
