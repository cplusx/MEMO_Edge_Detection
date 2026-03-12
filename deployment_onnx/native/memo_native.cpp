#include <torch/extension.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

template <typename scalar_t>
torch::Tensor local_maxima_mask_cpu_impl(const torch::Tensor& confidence, int64_t connectivity) {
    auto confidence_contiguous = confidence.contiguous();
    auto sizes = confidence_contiguous.sizes();
    const auto batch = sizes[0];
    const auto height = sizes[1];
    const auto width = sizes[2];

    auto output = torch::zeros({batch, height, width}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    const scalar_t* confidence_ptr = confidence_contiguous.data_ptr<scalar_t>();
    bool* output_ptr = output.data_ptr<bool>();

    auto idx = [height, width](int64_t batch_index, int64_t y, int64_t x) {
        return batch_index * height * width + y * width + x;
    };

    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                const auto center_index = idx(batch_index, y, x);
                const scalar_t center = confidence_ptr[center_index];
                bool is_max = true;

                for (int64_t dy = -1; dy <= 1 && is_max; ++dy) {
                    for (int64_t dx = -1; dx <= 1; ++dx) {
                        if (dy == 0 && dx == 0) {
                            continue;
                        }
                        if (connectivity == 4 && std::abs(dy) + std::abs(dx) != 1) {
                            continue;
                        }
                        const int64_t ny = y + dy;
                        const int64_t nx = x + dx;
                        if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                            continue;
                        }
                        const auto neighbor = confidence_ptr[idx(batch_index, ny, nx)];
                        if (center < neighbor) {
                            is_max = false;
                            break;
                        }
                    }
                }

                output_ptr[center_index] = is_max;
            }
        }
    }

    return output;
}

torch::Tensor local_maxima_mask_cpu(const torch::Tensor& confidence, int64_t connectivity) {
    TORCH_CHECK(confidence.device().is_cpu(), "local_maxima_mask_cpu expects a CPU tensor");
    TORCH_CHECK(confidence.dim() == 3, "confidence must have shape [B, H, W]");
    TORCH_CHECK(connectivity == 4 || connectivity == 8, "connectivity must be 4 or 8");

    torch::Tensor output;
    AT_DISPATCH_FLOATING_TYPES(confidence.scalar_type(), "local_maxima_mask_cpu", [&] {
        output = local_maxima_mask_cpu_impl<scalar_t>(confidence, connectivity);
    });
    return output;
}

torch::Tensor build_transfer_mask_cpu(
    const torch::Tensor& masked_edges,
    const torch::Tensor& confidence,
    double conf_thres,
    int64_t max_transfer,
    bool force_all_remaining,
    int64_t connectivity
) {
    TORCH_CHECK(masked_edges.device().is_cpu(), "masked_edges must be on CPU");
    TORCH_CHECK(confidence.device().is_cpu(), "confidence must be on CPU");
    TORCH_CHECK(masked_edges.dim() == 3, "masked_edges must have shape [B, H, W]");
    TORCH_CHECK(confidence.dim() == 3, "confidence must have shape [B, H, W]");
    TORCH_CHECK(masked_edges.sizes() == confidence.sizes(), "masked_edges and confidence must share the same shape");

    const auto batch = masked_edges.size(0);
    const auto height = masked_edges.size(1);
    const auto width = masked_edges.size(2);
    auto output = torch::zeros_like(masked_edges, torch::TensorOptions().dtype(torch::kBool));

    auto masked_edges_contiguous = masked_edges.contiguous();
    auto confidence_contiguous = confidence.contiguous();
    auto local_max = local_maxima_mask_cpu(confidence_contiguous, connectivity);

    const int64_t* masked_ptr = masked_edges_contiguous.data_ptr<int64_t>();
    const bool* local_max_ptr = local_max.data_ptr<bool>();
    bool* output_ptr = output.data_ptr<bool>();

    auto idx = [height, width](int64_t batch_index, int64_t y, int64_t x) {
        return batch_index * height * width + y * width + x;
    };

    AT_DISPATCH_FLOATING_TYPES(confidence_contiguous.scalar_type(), "build_transfer_mask_cpu", [&] {
        const scalar_t* confidence_ptr = confidence_contiguous.data_ptr<scalar_t>();
        for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
            std::vector<std::pair<scalar_t, int64_t>> candidates;
            candidates.reserve(height * width / 4);

            for (int64_t y = 0; y < height; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    const auto flat_index = idx(batch_index, y, x);
                    if (masked_ptr[flat_index] != 2) {
                        continue;
                    }
                    if (force_all_remaining) {
                        output_ptr[flat_index] = true;
                        continue;
                    }
                    if (!local_max_ptr[flat_index]) {
                        continue;
                    }
                    const scalar_t score = confidence_ptr[flat_index];
                    if (score <= static_cast<scalar_t>(conf_thres)) {
                        continue;
                    }
                    candidates.emplace_back(score, flat_index);
                }
            }

            if (force_all_remaining || candidates.empty()) {
                continue;
            }

            const auto keep = std::min<int64_t>(max_transfer, static_cast<int64_t>(candidates.size()));
            std::partial_sort(
                candidates.begin(),
                candidates.begin() + keep,
                candidates.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; }
            );

            for (int64_t index = 0; index < keep; ++index) {
                output_ptr[candidates[index].second] = true;
            }
        }
    });

    return output;
}

}  // namespace

#ifdef WITH_CUDA
torch::Tensor local_maxima_mask_cuda(const torch::Tensor& confidence, int64_t connectivity);
#endif

torch::Tensor local_maxima_mask_dispatch(const torch::Tensor& confidence, int64_t connectivity) {
#ifdef WITH_CUDA
    if (confidence.is_cuda()) {
        return local_maxima_mask_cuda(confidence, connectivity);
    }
#else
    TORCH_CHECK(!confidence.is_cuda(), "CUDA support was not enabled when building memo_native_ext");
#endif
    return local_maxima_mask_cpu(confidence, connectivity);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("local_maxima_mask", &local_maxima_mask_dispatch, "Local maxima mask (CPU/CUDA)");
    m.def("build_transfer_mask", &build_transfer_mask_cpu, "Build transfer mask (CPU)");
}