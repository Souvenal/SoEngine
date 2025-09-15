[shader("compute")]
[numthreads(16, 16, 1)]
void compMain(
    uint3  globalID   : SV_DispatchThreadID,
    uint3  localID    : SV_GroupThreadID,
    uint3  groupID    : SV_GroupID,
    [[vk::binding(0, 0)]] RWTexture2D<float4> image
)
{
    // 绑定：set = 0, binding = 0 (Vulkan); register(u0, space0) (DX)

    uint2 size;
    image.GetDimensions(size.x, size.y);

    if (globalID.x < size.x && globalID.y < size.y)
    {
        float4 color = float4(0.0, 0.0, 0.0, 1.0);

        if (localID.x != 0 && localID.y != 0)
        {
            color.x = float(globalID.x) / float(size.x);
            color.y = float(globalID.y) / float(size.y);
        }

        image[uint2(globalID.xy)] = color;
    }
}