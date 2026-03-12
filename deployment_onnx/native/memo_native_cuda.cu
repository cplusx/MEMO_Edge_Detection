#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void local_maxima_mask_kernel(
    const scalar_t* confidence,
    bool* output,
    int64_t batch,
    int64_t height,
    int64_t width,
    int64_t connectivity
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * height * width;
    if (index >= total) {
        return;
    }

    const int64_t batch_index = index / (height * width);
    const int64_t local_index = index % (height * width);
    const int64_t y = local_index / width;
    const int64_t x = local_index % width;
    const scalar_t center = confidence[index];

    bool is_max = true;
    for (int dy = -1; dy <= 1 && is_max; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dy == 0 && dx == 0) {
                continue;
            }
            if (connectivity == 4 && abs(dy) + abs(dx) != 1) {
                continue;
            }
            const int64_t ny = y + dy;
            const int64_t nx = x + dx;
            if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                continue;
            }
            const int64_t neighbor_index = batch_index * height * width + ny * width + nx;
            if (center < confidence[neighbor_index]) {
                is_max = false;
                break;
            }
        }
    }

    output[index] = is_max;
}

}  // namespace

torch::Tensor local_maxima_mask_cuda(const torch::Tensor& confidence, int64_t connectivity) {
    TORCH_CHECK(confidence.is_cuda(), "local_maxima_mask_cuda expects a CUDA tensor");
    TORCH_CHECK(confidence.dim() == 3, "confidence must have shape [B, H, W]");

    auto confidence_contiguous = confidence.contiguous();
    auto output = torch::zeros(
        confidence_contiguous.sizes(),
        torch::TensorOptions().dtype(torch::kBool).device(confidence_contiguous.device())
    );

    const auto batch = confidence_contiguous.size(0);
    const auto height = confidence_contiguous.size(1);
    const auto width = confidence_contiguous.size(2);
    const int64_t total = batch * height * width;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(confidence_contiguous.scalar_type(), "local_maxima_mask_cuda", [&] {
        local_maxima_mask_kernel<scalar_t><<<blocks, threads>>>(
            confidence_contiguous.data_ptr<scalar_t>(),
            output.data_ptr<bool>(),
            batch,
            height,
            width,
            connectivity
        );
    });

    return output;
}