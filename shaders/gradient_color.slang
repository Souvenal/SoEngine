[push_constant]
cbuffer ComputePushConstants {
    float4 data1;
    float4 data2;
    float4 data3;
    float4 data4;
} pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void compMain(
    uint3  globalID   : SV_DispatchThreadID,
    uint3  localID    : SV_GroupThreadID,
    uint3  groupID    : SV_GroupID,
    // 绑定：set = 0, binding = 0
    [[vk::binding(0, 0)]] RWTexture2D<float4> image,
)
{
    int2 texelCoord = int2(globalID.xy);
    
    uint2 size;
    image.GetDimensions(size.x, size.y);
    int2 imageSize = int2(size);

    float4 topColor = pc.data1;
    float4 bottomColor = pc.data2;

    if (texelCoord.x < imageSize.x && texelCoord.y < imageSize.y)
    {
        // if (localID.x != 0 && localID.y != 0) {
            float blend = float(texelCoord.y) / float(imageSize.y);
            
            float4 color = lerp(topColor, bottomColor, blend);
            image[uint2(texelCoord)] = color;
        // }
    }
}