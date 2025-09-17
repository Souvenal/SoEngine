struct Vertex {
    float3 position;
    float uvX;
    float3 normal; 
    float uvY;
    float4 color;
};

// Push constants for bindless rendering
[push_constant]
cbuffer PushConstants {
    float4x4 renderMatrix;
    Ptr<Vertex> vertexBuffer;
} pc;

struct VSOutput {
    float4 pos : SV_Position;
    float3 fragColor : COLOR0;
    float2 fragTexCoord : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID) {
    // Load vertex data from device address (bindless buffer access)
    // Vertex v = PushConstants.vertexBufferAddress + vertexID * sizeof(Vertex);
    Vertex v = loadAligned<16, Vertex>(pc.vertexBuffer + vertexID);
    
    VSOutput output;
    
    // Transform position
    output.pos = mul(pc.renderMatrix, float4(v.position, 1.0));
    
    // Pass through color and UV coordinates
    output.fragColor = v.color.xyz;
    output.fragTexCoord = float2(v.uvX, v.uvY);
    
    return output;
}

// // Texture and sampler for fragment shader
// Texture2D colorTexture;
// SamplerState texSampler;

// [shader("fragment")]
// float4 fragMain(VSOutput input) : SV_TARGET {
//     return colorTexture.Sample(texSampler, input.fragTexCoord) * float4(input.fragColor, 1.0);
// }

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_TARGET {
    return float4(input.fragColor, 1.0);
}