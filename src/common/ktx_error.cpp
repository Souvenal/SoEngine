#include "ktx_error.h"

#include <stdexcept>

void handleKtxError(KTX_error_code code) {
    switch (code) {
        case KTX_SUCCESS:
            // Success, no action needed.
            break;
        case KTX_INVALID_VALUE:
            throw std::runtime_error("KTX error: Invalid value (possibly incomplete parameters or callbacks, or newTex is NULL)");
        case KTX_FILE_DATA_ERROR:
            throw std::runtime_error("KTX error: File data error (source data is inconsistent with the KTX specification)");
        case KTX_FILE_READ_ERROR:
            throw std::runtime_error("KTX error: File read error (an error occurred while reading the source)");
        case KTX_FILE_UNEXPECTED_EOF:
            throw std::runtime_error("KTX error: Unexpected EOF (not enough data in the source)");
        case KTX_OUT_OF_MEMORY:
            throw std::runtime_error("KTX error: Out of memory (insufficient memory on CPU or Vulkan device)");
        case KTX_UNKNOWN_FILE_FORMAT:
            throw std::runtime_error("KTX error: Unknown file format (the source is not in KTX format)");
        case KTX_UNSUPPORTED_FEATURE:
            throw std::runtime_error("KTX error: Unsupported feature (sparse binding of KTX textures is not supported)");
        case KTX_UNSUPPORTED_TEXTURE_TYPE:
            throw std::runtime_error("KTX error: Unsupported texture type (the source describes a texture type not supported by OpenGL or Vulkan, e.g., a 3D array)");
        case KTX_INVALID_OPERATION:
            throw std::runtime_error("KTX error: Invalid operation (unsupported format, tiling, usageFlags, mipmap generation, or too many mip levels/layers)");
        default:
            throw std::runtime_error("KTX error: Unknown error code");
    }
}