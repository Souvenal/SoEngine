[push_constant]
cbuffer ComputePushConstants {
    float4 data1;
    float4 data2;
    float4 data3;
    float4 data4;
} pc;

// Return random noise in the range [0.0, 1.0], as a function of x.
float Noise2d(float2 x)
{
    float xhash = cos(x.x * 37.0);
    float yhash = cos(x.y * 57.0);
    return frac(415.92653 * (xhash + yhash));
}

// Convert Noise2d() into a "star field" by stomping everything below fThreshold to zero.
float NoisyStarField(float2 vSamplePos, float fThreshold)
{
    float StarVal = Noise2d(vSamplePos);
    if (StarVal >= fThreshold)
        StarVal = pow((StarVal - fThreshold) / (1.0 - fThreshold), 6.0);
    else
        StarVal = 0.0;
    return StarVal;
}

// Stabilize NoisyStarField() by only sampling at integer values.
float StableStarField(float2 vSamplePos, float fThreshold)
{
    // Linear interpolation between four samples.
    // Note: This approach has some visual artifacts.
    // There must be a better way to "anti alias" the star field.
    float fractX = frac(vSamplePos.x);
    float fractY = frac(vSamplePos.y);
    float2 floorSample = floor(vSamplePos);
    float v1 = NoisyStarField(floorSample, fThreshold);
    float v2 = NoisyStarField(floorSample + float2(0.0, 1.0), fThreshold);
    float v3 = NoisyStarField(floorSample + float2(1.0, 0.0), fThreshold);
    float v4 = NoisyStarField(floorSample + float2(1.0, 1.0), fThreshold);

    float StarVal = v1 * (1.0 - fractX) * (1.0 - fractY)
                  + v2 * (1.0 - fractX) * fractY
                  + v3 * fractX * (1.0 - fractY)
                  + v4 * fractX * fractY;
    return StarVal;
}

void mainImage(out float4 fragColor, float2 fragCoord, float2 iResolution)
{
    // Sky Background Color
    float3 vColor = pc.data1.xyz * fragCoord.y / iResolution.y;

    // Note: Choose fThreshold in the range [0.99, 0.9999].
    // Higher values (i.e., closer to one) yield a sparser starfield.
    float StarFieldThreshold = pc.data1.w; // 0.97;

    // Stars with a slow crawl.
    float xRate = 0.2;
    float yRate = -0.06;
    float2 vSamplePos = fragCoord.xy + float2(xRate * 1.0, yRate * 1.0);
    float StarVal = StableStarField(vSamplePos, StarFieldThreshold);
    vColor += float3(StarVal, StarVal, StarVal);
    
    fragColor = float4(vColor, 1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compMain(
    uint3 globalID : SV_DispatchThreadID,
    [[vk::binding(0, 0)]] RWTexture2D<float4> image
)
{
    int2 texelCoord = int2(globalID.xy);
    
    uint2 size;
    image.GetDimensions(size.x, size.y);
    float2 iResolution = float2(size);

    if (texelCoord.x < size.x && texelCoord.y < size.y)
    {
        float4 color;
        mainImage(color, float2(texelCoord), iResolution);
        
        image[uint2(texelCoord)] = color;
    }
}