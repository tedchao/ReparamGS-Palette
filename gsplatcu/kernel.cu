/* Copyright:
 * This file is part of gsplatcu.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#include "kernel.cuh"
#include "matrix.cuh"

#define MIN_DEPTH (0.2)
#define BAD_MARKER (-1.f)
#define MAX_SPLAT_DIM 8


// ─────────────────────────────────────────────────────────────────────────────
// Shared memory fetch helpers
// ─────────────────────────────────────────────────────────────────────────────

inline __device__ void fetch2shared_color(
    int32_t n,
    const bool is_backward,
    const int2 range,
    const int *__restrict__ gsid_per_patch,
    const float2 *__restrict__ us,
    const float3 *__restrict__ cinv2ds,
    const float *__restrict__ alphas,
    const float3 *__restrict__ colors,
    float2 *shared_pos2d,
    float3 *shared_cinv2d,
    float *shared_alpha,
    float3 *shared_color,
    int *shared_gsid)
{
    int i = blockDim.x * threadIdx.y + threadIdx.x;
    int j;
    if (is_backward)
        j = range.y - n * BLOCK_SIZE - i - 1;
    else
        j = range.x + n * BLOCK_SIZE + i;

    if (j < range.y && j >= range.x)
    {
        int gs_id = gsid_per_patch[j];
        shared_gsid[i]   = gs_id;
        shared_pos2d[i]  = us[gs_id];
        shared_cinv2d[i] = cinv2ds[gs_id];
        shared_alpha[i]  = alphas[gs_id];
        shared_color[i]  = colors[gs_id];
    }
}


inline __device__ void fetch2shared_weight(
    int32_t n,
    const bool is_backward,
    const int2 range,
    const int *__restrict__ gsid_per_patch,
    const float2 *__restrict__ us,
    const float3 *__restrict__ cinv2ds,
    const float *__restrict__ alphas,
    const float *__restrict__ weights,
    float2 *shared_pos2d,
    float3 *shared_cinv2d,
    float *shared_alpha,
    float *shared_weight,
    int *shared_gsid,
    const int num_palette)
{
    int splat_dim = num_palette;
    int i = blockDim.x * threadIdx.y + threadIdx.x;
    int j;
    if (is_backward)
        j = range.y - n * BLOCK_SIZE - i - 1;
    else
        j = range.x + n * BLOCK_SIZE + i;

    if (j < range.y && j >= range.x)
    {
        int gs_id = gsid_per_patch[j];
        shared_gsid[i]   = gs_id;
        shared_pos2d[i]  = us[gs_id];
        shared_cinv2d[i] = cinv2ds[gs_id];
        shared_alpha[i]  = alphas[gs_id];

        #pragma unroll
        for (int k = 0; k < splat_dim; ++k)
            shared_weight[i * splat_dim + k] = weights[gs_id * splat_dim + k];
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Tile/patch helpers
// ─────────────────────────────────────────────────────────────────────────────

__global__ void createKeys(
    const int gs_num,
    const float* __restrict__ depths,
    const uint32_t* __restrict__ patch_offset_per_gs,
    const uint4* __restrict__ rects,
    const dim3 grid,
    uint64_t* __restrict__ patch_keys,
    int* __restrict__ gsid_per_patch)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gs_num) return;

    const float depth = depths[i];
    if (depth < MIN_DEPTH) return;

    uint32_t off = (i == 0) ? 0 : patch_offset_per_gs[i - 1];
    uint4 rect = rects[i];

    for (uint y = rect.y; y < rect.w; y++)
        for (uint x = rect.x; x < rect.z; x++)
        {
            uint64_t key = (y * grid.x + x);
            key <<= 32;
            key |= (uint32_t)(depth * 1000);
            patch_keys[off] = key;
            gsid_per_patch[off] = i;
            off++;
        }
}


__global__ void getRects(
    const int gs_num,
    const float* __restrict__ us,
    int2* __restrict__ areas,
    float* __restrict__ depths,
    const dim3 grid,
    uint4 *__restrict__ gs_rects,
    uint* __restrict__ patch_num_per_gs)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gs_num) return;

    patch_num_per_gs[idx] = 0;
    if (depths[idx] < MIN_DEPTH) return;

    float2 u = {us[2 * idx], us[2 * idx + 1]};
    float xs = areas[idx].x;
    float ys = areas[idx].y;

    uint4 rect = {
        min(grid.x, max((int)0, (int)((u.x - xs) / BLOCK))),
        min(grid.y, max((int)0, (int)((u.y - ys) / BLOCK))),
        min(grid.x, max((int)0, (int)(DIV_ROUND_UP(u.x + xs, BLOCK)))),
        min(grid.y, max((int)0, (int)(DIV_ROUND_UP(u.y + ys, BLOCK))))
    };

    uint n = (rect.w - rect.y) * (rect.z - rect.x);
    if (n == 0)
    {
        depths[idx] = BAD_MARKER;
        areas[idx]  = {0, 0};
        return;
    }
    gs_rects[idx]          = rect;
    patch_num_per_gs[idx]  = n;
}


__global__ void getRanges(
    const int patch_num,
    const uint64_t *__restrict__ patch_keys,
    int2 *__restrict__ patch_range_per_tile)
{
    const int cur_patch = blockIdx.x * blockDim.x + threadIdx.x;
    if (cur_patch >= patch_num) return;

    const int prv_patch = cur_patch == 0 ? 0 : cur_patch - 1;
    uint32_t cur_tile = patch_keys[cur_patch] >> 32;
    uint32_t prv_tile = patch_keys[prv_patch] >> 32;

    if (cur_patch == 0)
        patch_range_per_tile[cur_tile].x = 0;
    else if (cur_patch == patch_num - 1)
        patch_range_per_tile[cur_tile].y = patch_num;

    if (prv_tile != cur_tile)
    {
        patch_range_per_tile[prv_tile].y = cur_patch;
        patch_range_per_tile[cur_tile].x = cur_patch;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Forward: color splatting
// ─────────────────────────────────────────────────────────────────────────────

__global__ void draw_color __launch_bounds__(BLOCK * BLOCK)(
    const int width,
    const int height,
    const int2 *__restrict__ patch_range_per_tile,
    const int *__restrict__ gsid_per_patch,
    const float2 *__restrict__ us,
    const float3 *__restrict__ cinv2ds,
    const float *__restrict__ alphas,
    const float3 *__restrict__ colors,
    float *__restrict__ image,
    int *__restrict__ contrib,
    float *__restrict__ final_tau)
{
    const uint2 tile = {blockIdx.x, blockIdx.y};
    const uint2 pix  = {tile.x * BLOCK + threadIdx.x,
                        tile.y * BLOCK + threadIdx.y};

    const int      tile_idx = tile.y * gridDim.x + tile.x;
    const uint32_t pix_idx  = width * pix.y + pix.x;

    const bool inside = pix.x < width && pix.y < height;
    const int2 range  = patch_range_per_tile[tile_idx];
    const int  gs_num = range.y - range.x;
    if (gs_num == 0) return;

    bool thread_is_finished = !inside;

    __shared__ float2 shared_pos2d [BLOCK_SIZE];
    __shared__ float3 shared_cinv2d[BLOCK_SIZE];
    __shared__ float  shared_alpha [BLOCK_SIZE];
    __shared__ float3 shared_color [BLOCK_SIZE];
    __shared__ int    shared_gsid  [BLOCK_SIZE];

    float3 final_color = {0.f, 0.f, 0.f};
    float  tau         = 1.0f;
    int    cont        = 0;
    int    cont_tmp    = 0;

    for (int i = 0; i < gs_num; i++)
    {
        int finished = __syncthreads_count(thread_is_finished);
        if (finished == BLOCK_SIZE) break;

        int j = i % BLOCK_SIZE;
        if (j == 0)
        {
            fetch2shared_color(i / BLOCK_SIZE, false, range,
                               gsid_per_patch, us, cinv2ds, alphas, colors,
                               shared_pos2d, shared_cinv2d, shared_alpha,
                               shared_color, shared_gsid);
            __syncthreads();
        }

        if (thread_is_finished) continue;

        float2 u    = shared_pos2d[j];
        float3 cinv = shared_cinv2d[j];
        float  alpha = shared_alpha[j];
        float3 color = shared_color[j];
        float2 d    = u - pix;

        cont_tmp++;

        float maha_dist  = max(0.0f, mahaSqDist(cinv, d));
        float alpha_prime = min(0.99f, alpha * exp(-0.5f * maha_dist));
        if (alpha_prime < 0.002f) continue;

        final_color += tau * alpha_prime * color;
        cont = cont_tmp;

        tau *= (1.f - alpha_prime);
        if (tau < 0.0001f) { thread_is_finished = true; continue; }
    }

    if (inside)
    {
        image[height * width * 0 + pix_idx] = final_color.x;
        image[height * width * 1 + pix_idx] = final_color.y;
        image[height * width * 2 + pix_idx] = final_color.z;
        contrib[pix_idx]   = cont;
        final_tau[pix_idx] = tau;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Forward: weight splatting
// ─────────────────────────────────────────────────────────────────────────────

__global__ void draw_weight __launch_bounds__(BLOCK * BLOCK)(
    const int width,
    const int height,
    const int2 *__restrict__ patch_range_per_tile,
    const int *__restrict__ gsid_per_patch,
    const float2 *__restrict__ us,
    const float3 *__restrict__ cinv2ds,
    const float *__restrict__ alphas,
    const float *__restrict__ weights,
    float *__restrict__ layer,
    int *__restrict__ contrib,
    float *__restrict__ final_tau,
    const int num_palette)
{
    int splat_dim = num_palette;

    const uint2 tile = {blockIdx.x, blockIdx.y};
    const uint2 pix  = {tile.x * BLOCK + threadIdx.x,
                        tile.y * BLOCK + threadIdx.y};

    const int      tile_idx = tile.y * gridDim.x + tile.x;
    const uint32_t pix_idx  = width * pix.y + pix.x;

    const bool inside = pix.x < width && pix.y < height;
    const int2 range  = patch_range_per_tile[tile_idx];
    const int  gs_num = range.y - range.x;
    if (gs_num == 0) return;

    bool thread_is_finished = !inside;

    __shared__ float2 shared_pos2d [BLOCK_SIZE];
    __shared__ float3 shared_cinv2d[BLOCK_SIZE];
    __shared__ float  shared_alpha [BLOCK_SIZE];
    __shared__ float  shared_weight[BLOCK_SIZE * MAX_SPLAT_DIM];
    __shared__ int    shared_gsid  [BLOCK_SIZE];

    float final_weight[MAX_SPLAT_DIM];
    for (int k = 0; k < splat_dim; ++k) final_weight[k] = 0.f;

    float tau      = 1.0f;
    int   cont     = 0;
    int   cont_tmp = 0;

    for (int i = 0; i < gs_num; i++)
    {
        int finished = __syncthreads_count(thread_is_finished);
        if (finished == BLOCK_SIZE) break;

        int j = i % BLOCK_SIZE;
        if (j == 0)
        {
            fetch2shared_weight(i / BLOCK_SIZE, false, range,
                                gsid_per_patch, us, cinv2ds, alphas, weights,
                                shared_pos2d, shared_cinv2d, shared_alpha,
                                shared_weight, shared_gsid, num_palette);
            __syncthreads();
        }

        if (thread_is_finished) continue;

        float2 u     = shared_pos2d[j];
        float3 cinv  = shared_cinv2d[j];
        float  alpha = shared_alpha[j];
        float *w     = &shared_weight[j * splat_dim];
        float2 d     = u - pix;

        cont_tmp++;

        float maha_dist   = max(0.0f, mahaSqDist(cinv, d));
        float alpha_prime = min(0.99f, alpha * exp(-0.5f * maha_dist));
        if (alpha_prime < 0.002f) continue;

        #pragma unroll
        for (int k = 0; k < splat_dim; ++k)
            final_weight[k] += tau * alpha_prime * w[k];
        cont = cont_tmp;

        tau *= (1.f - alpha_prime);
        if (tau < 0.0001f) { thread_is_finished = true; continue; }
    }

    if (inside)
    {
        #pragma unroll
        for (int k = 0; k < splat_dim; ++k)
            layer[k * height * width + pix_idx] = final_weight[k];
        contrib[pix_idx]   = cont;
        final_tau[pix_idx] = tau;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// inverseCov2D
// ─────────────────────────────────────────────────────────────────────────────

__global__ void inverseCov2D(
    int gs_num,
    const float3 *__restrict__ cov2ds,
    float *__restrict__ depths,
    float3 *__restrict__ cinv2ds,
    int2 *__restrict__ areas,
    float *__restrict__ dcinv2d_dcov2ds)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gs_num) return;
    if (depths[i] < MIN_DEPTH) return;

    const float3 cov2d = cov2ds[i];
    const float a = cov2d.x;
    const float b = cov2d.y;
    const float c = cov2d.z;

    const float det_inv = 1.f / (a * c - b * b);
    if (isnan(det_inv)) { depths[i] = BAD_MARKER; return; }

    cinv2ds[i] = {det_inv * c, -det_inv * b, det_inv * a};
    areas[i]   = {(int)ceil(3 * sqrt(abs(a))), (int)ceil(3 * sqrt(abs(c)))};

    if (dcinv2d_dcov2ds != nullptr)
    {
        const float det2_inv = det_inv * det_inv;
        dcinv2d_dcov2ds[i * 9 + 0] = -c * c * det2_inv;
        dcinv2d_dcov2ds[i * 9 + 1] =  2 * b * c * det2_inv;
        dcinv2d_dcov2ds[i * 9 + 2] = -a * c * det2_inv + det_inv;
        dcinv2d_dcov2ds[i * 9 + 3] =  b * c * det2_inv;
        dcinv2d_dcov2ds[i * 9 + 4] = -2 * b * b * det2_inv - det_inv;
        dcinv2d_dcov2ds[i * 9 + 5] =  a * b * det2_inv;
        dcinv2d_dcov2ds[i * 9 + 6] = -a * c * det2_inv + det_inv;
        dcinv2d_dcov2ds[i * 9 + 7] =  2 * a * b * det2_inv;
        dcinv2d_dcov2ds[i * 9 + 8] = -a * a * det2_inv;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// computeCov3D
// ─────────────────────────────────────────────────────────────────────────────

__global__ void computeCov3D(
    int32_t gs_num,
    const float4 *__restrict__ rots,
    const float3 *__restrict__ scales,
    const float *__restrict__ depths,
    float *__restrict__ cov3ds,
    float *__restrict__ dcov3d_drots,
    float *__restrict__ dcov3d_dscales)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gs_num) return;
    if (depths[i] < MIN_DEPTH) return;

    const float4 rot = rots[i];
    const float3 s   = scales[i];
    const float w = rot.x, x = rot.y, y = rot.z, z = rot.w;
    const float s0 = s.x,  s1 = s.y,  s2 = s.z;

    const float xx = x*x, yy = y*y, zz = z*z;
    const float xy = x*y, xz = x*z, yz = y*z;
    const float xw = x*w, yw = y*w, zw = z*w;

    Matrix<3,3> R = {
        1.f-2.f*(yy+zz), 2.f*(xy-zw),     2.f*(xz+yw),
        2.f*(xy+zw),     1.f-2.f*(xx+zz), 2.f*(yz-xw),
        2.f*(xz-yw),     2.f*(yz+xw),     1.f-2.f*(xx+yy)};

    Matrix<3,3> M = {
        R(0,0)*s0, R(0,1)*s1, R(0,2)*s2,
        R(1,0)*s0, R(1,1)*s1, R(1,2)*s2,
        R(2,0)*s0, R(2,1)*s1, R(2,2)*s2};

    Matrix<3,3> Sigma = M * M.transpose();

    cov3ds[i*6+0] = Sigma(0,0);
    cov3ds[i*6+1] = Sigma(0,1);
    cov3ds[i*6+2] = Sigma(0,2);
    cov3ds[i*6+3] = Sigma(1,1);
    cov3ds[i*6+4] = Sigma(1,2);
    cov3ds[i*6+5] = Sigma(2,2);

    if (dcov3d_drots != nullptr && dcov3d_dscales != nullptr)
    {
        Matrix<6,9> dcov3d_dm = {
            2*M(0,0), 2*M(0,1), 2*M(0,2), 0,        0,        0,        0,        0,        0,
            M(1,0),  M(1,1),  M(1,2),  M(0,0),  M(0,1),  M(0,2),  0,        0,        0,
            M(2,0),  M(2,1),  M(2,2),  0,        0,        0,        M(0,0),  M(0,1),  M(0,2),
            0,        0,        0,        2*M(1,0), 2*M(1,1), 2*M(1,2), 0,        0,        0,
            0,        0,        0,        M(2,0),  M(2,1),  M(2,2),  M(1,0),  M(1,1),  M(1,2),
            0,        0,        0,        0,        0,        0,        2*M(2,0), 2*M(2,1), 2*M(2,2)};

        Matrix<9,4> dm_rot = {
            0,           0,           -4*s0*y,    -4*s0*z,
            -2*s1*z,     2*s1*y,      2*s1*x,     -2*s1*w,
            2*s2*y,      2*s2*z,      2*s2*w,     2*s2*x,
            2*s0*z,      2*s0*y,      2*s0*x,     2*s0*w,
            0,           -4*s1*x,     0,           -4*s1*z,
            -2*s2*x,    -2*s2*w,     2*s2*z,      2*s2*y,
            -2*s0*y,     2*s0*z,     -2*s0*w,     2*s0*x,
            2*s1*x,      2*s1*w,      2*s1*z,     2*s1*y,
            0,           -4*s2*x,    -4*s2*y,     0};

        Matrix<9,3> dm_scale = {
            R(0,0), 0,      0,
            0,      R(0,1), 0,
            0,      0,      R(0,2),
            R(1,0), 0,      0,
            0,      R(1,1), 0,
            0,      0,      R(1,2),
            R(2,0), 0,      0,
            0,      R(2,1), 0,
            0,      0,      R(2,2)};

        Matrix<6,4> dcov3d_drot   = dcov3d_dm * dm_rot;
        Matrix<6,3> dcov3d_dscale = dcov3d_dm * dm_scale;

        for (int j = 0; j < 24; j++) dcov3d_drots  [i*24 + j] = dcov3d_drot(j);
        for (int j = 0; j < 18; j++) dcov3d_dscales[i*18 + j] = dcov3d_dscale(j);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// computeCov2D
// ─────────────────────────────────────────────────────────────────────────────

__global__ void computeCov2D(
    int32_t gs_num,
    const float *__restrict__ cov3ds,
    const float3 *__restrict__ pcs,
    const float *__restrict__ Rcw,
    const float *__restrict__ depths,
    const float focal_x,
    const float focal_y,
    const float tan_fovx,
    const float tan_fovy,
    float3 *__restrict__ cov2ds,
    float *__restrict__ dcov2d_dcov3ds,
    float *__restrict__ dcov2d_dpcs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gs_num) return;
    if (depths[i] < MIN_DEPTH) return;

    float3 pc = pcs[i];
    float x = pc.x, y = pc.y;
    const float z = pc.z;

    const float a = cov3ds[i*6+0], b = cov3ds[i*6+1], c = cov3ds[i*6+2];
    const float d = cov3ds[i*6+3], e = cov3ds[i*6+4], f = cov3ds[i*6+5];

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    x = min(limx, max(-limx, x / z)) * z;
    y = min(limy, max(-limy, y / z)) * z;

    const float z2 = z * z;

    Matrix<3,3> R = {
        Rcw[0], Rcw[1], Rcw[2],
        Rcw[3], Rcw[4], Rcw[5],
        Rcw[6], Rcw[7], Rcw[8]};

    Matrix<2,3> J = {
        focal_x/z, 0.0f,      -(focal_x*x)/z2,
        0.0f,      focal_y/z, -(focal_y*y)/z2};

    Matrix<3,3> Sigma = {a, b, c, b, d, e, c, e, f};
    Matrix<2,3> M     = J * R;
    Matrix<2,2> Sigma_prime = M * Sigma * M.transpose();

    cov2ds[i] = {Sigma_prime(0,0) + 0.3f,
                 Sigma_prime(0,1),
                 Sigma_prime(1,1) + 0.3f};

    if (dcov2d_dcov3ds != nullptr && dcov2d_dpcs != nullptr)
    {
        Matrix<3,6> dcov2d_dcov3d = {
            M(0,0)*M(0,0),
            2*M(0,0)*M(0,1),
            2*M(0,0)*M(0,2),
            M(0,1)*M(0,1),
            2*M(0,1)*M(0,2),
            M(0,2)*M(0,2),
            M(0,0)*M(1,0),
            M(0,0)*M(1,1)+M(0,1)*M(1,0),
            M(0,0)*M(1,2)+M(0,2)*M(1,0),
            M(0,1)*M(1,1),
            M(0,1)*M(1,2)+M(0,2)*M(1,1),
            M(0,2)*M(1,2),
            M(1,0)*M(1,0),
            2*M(1,0)*M(1,1),
            2*M(1,0)*M(1,2),
            M(1,1)*M(1,1),
            2*M(1,1)*M(1,2),
            M(1,2)*M(1,2)};

        Matrix<3,6> dcov2d_dm = {
            2*a*M(0,0)+2*b*M(0,1)+2*c*M(0,2),
            2*b*M(0,0)+2*d*M(0,1)+2*e*M(0,2),
            2*c*M(0,0)+2*e*M(0,1)+2*f*M(0,2),
            0, 0, 0,
            a*M(1,0)+b*M(1,1)+c*M(1,2),
            b*M(1,0)+d*M(1,1)+e*M(1,2),
            c*M(1,0)+e*M(1,1)+f*M(1,2),
            a*M(0,0)+b*M(0,1)+c*M(0,2),
            b*M(0,0)+d*M(0,1)+e*M(0,2),
            c*M(0,0)+e*M(0,1)+f*M(0,2),
            0, 0, 0,
            2*a*M(1,0)+2*b*M(1,1)+2*c*M(1,2),
            2*b*M(1,0)+2*d*M(1,1)+2*e*M(1,2),
            2*c*M(1,0)+2*e*M(1,1)+2*f*M(1,2)};

        const float z2_inv = 1.f / (z*z);
        const float z3_inv = z2_inv / z;

        Matrix<6,3> dm_dpc = {
            -focal_x*R(2,0)*z2_inv, 0, -focal_x*R(0,0)*z2_inv + 2*focal_x*R(2,0)*x*z3_inv,
            -focal_x*R(2,1)*z2_inv, 0, -focal_x*R(0,1)*z2_inv + 2*focal_x*R(2,1)*x*z3_inv,
            -focal_x*R(2,2)*z2_inv, 0, -focal_x*R(0,2)*z2_inv + 2*focal_x*R(2,2)*x*z3_inv,
            0, -focal_y*R(2,0)*z2_inv, -focal_y*R(1,0)*z2_inv + 2*focal_y*R(2,0)*y*z3_inv,
            0, -focal_y*R(2,1)*z2_inv, -focal_y*R(1,1)*z2_inv + 2*focal_y*R(2,1)*y*z3_inv,
            0, -focal_y*R(2,2)*z2_inv, -focal_y*R(1,2)*z2_inv + 2*focal_y*R(2,2)*y*z3_inv};

        Matrix<3,3> dcov2d_dpc = dcov2d_dm * dm_dpc;

        for (int j = 0; j < 9;  j++) dcov2d_dpcs  [i*9  + j] = dcov2d_dpc(j);
        for (int j = 0; j < 18; j++) dcov2d_dcov3ds[i*18 + j] = dcov2d_dcov3d(j);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// project
// ─────────────────────────────────────────────────────────────────────────────

__global__ void project(
    int32_t gs_num,
    const float3 *__restrict__ pws,
    const float *__restrict__ Rcw,
    const float3 *__restrict__ tcw,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    float2 *__restrict__ us,
    float3 *__restrict__ pcs,
    float *__restrict__ depths,
    float *__restrict__ du_dpcs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gs_num) return;

    Matrix<3,1> pw = {pws[i].x, pws[i].y, pws[i].z};
    Matrix<3,1> t  = {tcw[0].x, tcw[0].y, tcw[0].z};
    Matrix<3,3> R  = {
        Rcw[0], Rcw[1], Rcw[2],
        Rcw[3], Rcw[4], Rcw[5],
        Rcw[6], Rcw[7], Rcw[8]};

    Matrix<3,1> pc = R * pw + t;
    const float x = pc(0), y = pc(1), z = pc(2);

    if (z < MIN_DEPTH) { depths[i] = BAD_MARKER; return; }

    const float z_inv  = 1.f / z;
    const float z2_inv = z_inv * z_inv;
    const float x_fx   = x * focal_x;
    const float y_fy   = y * focal_y;

    us[i]     = {x_fx * z_inv + center_x, y_fy * z_inv + center_y};
    pcs[i]    = {x, y, z};
    depths[i] = z;

    if (du_dpcs != nullptr)
    {
        du_dpcs[i*6 + 0] =  focal_x * z_inv;
        du_dpcs[i*6 + 2] = -x_fx    * z2_inv;
        du_dpcs[i*6 + 4] =  focal_y * z_inv;
        du_dpcs[i*6 + 5] = -y_fy    * z2_inv;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// sh2Weight
// ─────────────────────────────────────────────────────────────────────────────

__global__ void sh2Weight(
    int32_t gs_num,
    const float *__restrict__ shs,
    const float3 *__restrict__ pws,
    const float3 *__restrict__ twc,
    const int sh_dim3,
    const int num_palette,
    float *__restrict__ weights,
    float *__restrict__ dweight_dshs,
    float *__restrict__ dweight_dpws)
{
    int splat_dim = num_palette;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gs_num) return;

    float dwl_dsh1,  dwl_dsh2,  dwl_dsh3;
    float dwl_dsh4,  dwl_dsh5,  dwl_dsh6,  dwl_dsh7,  dwl_dsh8;
    float dwl_dsh9,  dwl_dsh10, dwl_dsh11, dwl_dsh12, dwl_dsh13, dwl_dsh14, dwl_dsh15;
    float d0, d1, d2, normd_inv;
    float x, y, z, xx, yy, zz, xy, xz, yz;

    float weight[MAX_SPLAT_DIM];
    for (int k = 0; k < splat_dim; ++k) weight[k] = 0.f;

    const float *sh_base = shs + i * sh_dim3 * splat_dim;
    const float *sh0  = sh_base +  0*splat_dim;
    const float *sh1  = sh_base +  1*splat_dim;
    const float *sh2  = sh_base +  2*splat_dim;
    const float *sh3  = sh_base +  3*splat_dim;
    const float *sh4  = sh_base +  4*splat_dim;
    const float *sh5  = sh_base +  5*splat_dim;
    const float *sh6  = sh_base +  6*splat_dim;
    const float *sh7  = sh_base +  7*splat_dim;
    const float *sh8  = sh_base +  8*splat_dim;
    const float *sh9  = sh_base +  9*splat_dim;
    const float *sh10 = sh_base + 10*splat_dim;
    const float *sh11 = sh_base + 11*splat_dim;
    const float *sh12 = sh_base + 12*splat_dim;
    const float *sh13 = sh_base + 13*splat_dim;
    const float *sh14 = sh_base + 14*splat_dim;
    const float *sh15 = sh_base + 15*splat_dim;

    // Level 0
    #pragma unroll
    for (int k = 0; k < splat_dim; ++k) weight[k] += SH_C0_0 * sh0[k];

    if (sh_dim3 > 1)
    {
        float3 dv = pws[i] - twc[0];
        d0 = dv.x; d1 = dv.y; d2 = dv.z;
        normd_inv = 1.f / sqrt(d0*d0 + d1*d1 + d2*d2);
        float3 dir = dv * normd_inv;
        x = dir.x; y = dir.y; z = dir.z;

        dwl_dsh1 = SH_C1_0 * y;
        dwl_dsh2 = SH_C1_1 * z;
        dwl_dsh3 = SH_C1_2 * x;

        #pragma unroll
        for (int k = 0; k < splat_dim; ++k)
            weight[k] += dwl_dsh1*sh1[k] + dwl_dsh2*sh2[k] + dwl_dsh3*sh3[k];

        if (sh_dim3 > 4)
        {
            xx=x*x; yy=y*y; zz=z*z; xy=x*y; yz=y*z; xz=x*z;
            dwl_dsh4 = SH_C2_0 * xy;
            dwl_dsh5 = SH_C2_1 * yz;
            dwl_dsh6 = SH_C2_2 * (2.f*zz - xx - yy);
            dwl_dsh7 = SH_C2_3 * xz;
            dwl_dsh8 = SH_C2_4 * (xx - yy);

            #pragma unroll
            for (int k = 0; k < splat_dim; ++k)
                weight[k] += dwl_dsh4*sh4[k] + dwl_dsh5*sh5[k] + dwl_dsh6*sh6[k]
                           + dwl_dsh7*sh7[k] + dwl_dsh8*sh8[k];

            if (sh_dim3 > 9)
            {
                dwl_dsh9  = SH_C3_0 * y * (3.f*xx - yy);
                dwl_dsh10 = SH_C3_1 * xy * z;
                dwl_dsh11 = SH_C3_2 * y * (4.f*zz - xx - yy);
                dwl_dsh12 = SH_C3_3 * z * (2.f*zz - 3.f*xx - 3.f*yy);
                dwl_dsh13 = SH_C3_4 * x * (4.f*zz - xx - yy);
                dwl_dsh14 = SH_C3_5 * z * (xx - yy);
                dwl_dsh15 = SH_C3_6 * x * (xx - 3.f*yy);

                #pragma unroll
                for (int k = 0; k < splat_dim; ++k)
                    weight[k] += dwl_dsh9*sh9[k]   + dwl_dsh10*sh10[k] + dwl_dsh11*sh11[k]
                               + dwl_dsh12*sh12[k] + dwl_dsh13*sh13[k] + dwl_dsh14*sh14[k]
                               + dwl_dsh15*sh15[k];
            }
        }
    }

    float *w_out = weights + i * splat_dim;
    #pragma unroll
    for (int k = 0; k < splat_dim; ++k) w_out[k] = weight[k];

    if (dweight_dshs != nullptr && dweight_dpws != nullptr)
    {
        float dw_dr0[MAX_SPLAT_DIM], dw_dr1[MAX_SPLAT_DIM], dw_dr2[MAX_SPLAT_DIM];
        for (int k = 0; k < splat_dim; ++k) { dw_dr0[k]=0.f; dw_dr1[k]=0.f; dw_dr2[k]=0.f; }

        Matrix<3,3> dr_dpw = {0,0,0, 0,0,0, 0,0,0};

        dweight_dshs[sh_dim3 * i + 0] = SH_C0_0;

        if (sh_dim3 > 1)
        {
            const float normd3_inv = normd_inv * normd_inv * normd_inv;
            const float dr00 = -d0*d0*normd3_inv + normd_inv;
            const float dr11 = -d1*d1*normd3_inv + normd_inv;
            const float dr22 = -d2*d2*normd3_inv + normd_inv;
            const float dr01 = -d0*d1*normd3_inv;
            const float dr02 = -d0*d2*normd3_inv;
            const float dr12 = -d1*d2*normd3_inv;
            dr_dpw = {dr00, dr01, dr02, dr01, dr11, dr12, dr02, dr12, dr22};

            dweight_dshs[sh_dim3*i+1] = dwl_dsh1;
            dweight_dshs[sh_dim3*i+2] = dwl_dsh2;
            dweight_dshs[sh_dim3*i+3] = dwl_dsh3;

            #pragma unroll
            for (int k = 0; k < splat_dim; ++k)
            {
                dw_dr0[k] += SH_C1_2 * sh3[k];
                dw_dr1[k] += SH_C1_0 * sh1[k];
                dw_dr2[k] += SH_C1_1 * sh2[k];
            }

            if (sh_dim3 > 4)
            {
                dweight_dshs[sh_dim3*i+4] = dwl_dsh4;
                dweight_dshs[sh_dim3*i+5] = dwl_dsh5;
                dweight_dshs[sh_dim3*i+6] = dwl_dsh6;
                dweight_dshs[sh_dim3*i+7] = dwl_dsh7;
                dweight_dshs[sh_dim3*i+8] = dwl_dsh8;

                #pragma unroll
                for (int k = 0; k < splat_dim; ++k)
                {
                    dw_dr0[k] += SH_C2_0*y*sh4[k] - SH_C2_2*2*x*sh6[k]
                               + SH_C2_3*z*sh7[k]  + SH_C2_4*2*x*sh8[k];
                    dw_dr1[k] += SH_C2_0*x*sh4[k]  + SH_C2_1*z*sh5[k]
                               - SH_C2_2*2*y*sh6[k] - SH_C2_4*2*y*sh8[k];
                    dw_dr2[k] += SH_C2_1*y*sh5[k]  + SH_C2_2*4.f*z*sh6[k]
                               + SH_C2_3*x*sh7[k];
                }

                if (sh_dim3 > 9)
                {
                    dweight_dshs[sh_dim3*i+ 9] = dwl_dsh9;
                    dweight_dshs[sh_dim3*i+10] = dwl_dsh10;
                    dweight_dshs[sh_dim3*i+11] = dwl_dsh11;
                    dweight_dshs[sh_dim3*i+12] = dwl_dsh12;
                    dweight_dshs[sh_dim3*i+13] = dwl_dsh13;
                    dweight_dshs[sh_dim3*i+14] = dwl_dsh14;
                    dweight_dshs[sh_dim3*i+15] = dwl_dsh15;

                    #pragma unroll
                    for (int k = 0; k < splat_dim; ++k)
                    {
                        dw_dr0[k] += 6.f*SH_C3_0*x*y*sh9[k]
                                   + SH_C3_1*y*z*sh10[k]
                                   - 2.f*SH_C3_2*x*y*sh11[k]
                                   - 6.f*SH_C3_3*x*z*sh12[k]
                                   + SH_C3_4*sh13[k]*(4.f*zz-3.f*xx-yy)
                                   + 2.f*SH_C3_5*x*z*sh14[k]
                                   + SH_C3_6*sh15[k]*(3.f*xx-3.f*yy);
                        dw_dr1[k] += SH_C3_0*sh9[k]*(3.f*xx-2.f*yy)
                                   + SH_C3_1*x*z*sh10[k]
                                   + SH_C3_2*sh11[k]*(4.f*zz-xx-3.f*yy)
                                   - 6.f*SH_C3_3*y*z*sh12[k]
                                   - 2.f*SH_C3_4*x*y*sh13[k]
                                   - 2.f*SH_C3_5*y*z*sh14[k]
                                   - 6.f*SH_C3_6*x*y*sh15[k];
                        dw_dr2[k] += SH_C3_1*x*y*sh10[k]
                                   + 8.f*SH_C3_2*y*z*sh11[k]
                                   + SH_C3_3*sh12[k]*(6.f*zz-3.f*xx-3.f*yy)
                                   + 8.f*SH_C3_4*x*z*sh13[k]
                                   + SH_C3_5*sh14[k]*(xx-yy);
                    }
                }
            }
        }

        Matrix<MAX_SPLAT_DIM,3> dw_dpw;
        for (int k = 0; k < splat_dim; ++k)
        {
            dw_dpw(k,0) = dw_dr0[k]*dr_dpw(0,0) + dw_dr1[k]*dr_dpw(1,0) + dw_dr2[k]*dr_dpw(2,0);
            dw_dpw(k,1) = dw_dr0[k]*dr_dpw(0,1) + dw_dr1[k]*dr_dpw(1,1) + dw_dr2[k]*dr_dpw(2,1);
            dw_dpw(k,2) = dw_dr0[k]*dr_dpw(0,2) + dw_dr1[k]*dr_dpw(1,2) + dw_dr2[k]*dr_dpw(2,2);
        }

        const int offset = i * splat_dim * 3;
        for (int k = 0; k < splat_dim; ++k)
            for (int j = 0; j < 3; ++j)
                dweight_dpws[offset + k*3 + j] = dw_dpw(k, j);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// sh2Weight_fast  (forward, rank=1)
//
// General P x Q reshape of the KL coefficient block, rank fixed to 1.
// shs_A: (N, P)   — factor vector a_i in R^P
// shs_B: (N, Q)   — factor vector b_i in R^Q
// C_i = a_i * b_i^T flattened channel-major, read as band-major by kernel.
// Index arithmetic: g = lm*K + k, row = g/Q, col = g%Q
// weight[k] += sum_lm  A[row] * B[col] * Ylm[lm]
// ─────────────────────────────────────────────────────────────────────────────

__global__ void sh2Weight_fast(
    int32_t gs_num,
    const float *__restrict__ shs_dc,      // (N, K)
    const float *__restrict__ shs_A,       // (N, P)
    const float *__restrict__ shs_B,       // (N, Q)
    const float3 *__restrict__ pws,
    const float3 *__restrict__ twc,
    const int sh_dim3,                     // L + 1
    const int num_palette,                 // K
    const int Q,                           // row-block length
    float *__restrict__ weights,           // (N, K)
    float *__restrict__ dweight_ddc,       // (N, K)
    float *__restrict__ Ylm_stored         // (N, L)
)
{
    const int K = num_palette;
    const int L = sh_dim3 - 1;
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gs_num) return;
    
    // ── DC contribution ───────────────────────────────────────────────────────
    const float *dc = shs_dc + i * K;
    float weight[MAX_SPLAT_DIM];
    for (int k = 0; k < K; ++k)
        {
            weight[k]              = SH_C0_0 * dc[k];
            dweight_ddc[i * K + k] = SH_C0_0;
        }
    
    if (sh_dim3 <= 1)
        {
            for (int k = 0; k < K; ++k) weights[i * K + k] = weight[k];
            return;
        }
    
    // ── View direction ────────────────────────────────────────────────────────
    float d0, d1, d2, normd_inv;
    float x, y, z, xx, yy, zz, xy, xz, yz;
    
    float3 dv  = pws[i] - twc[0];
    d0 = dv.x; d1 = dv.y; d2 = dv.z;
    normd_inv  = 1.f / sqrtf(d0*d0 + d1*d1 + d2*d2);
    float3 dir = dv * normd_inv;
    x = dir.x; y = dir.y; z = dir.z;
    
    // ── Evaluate Y_lm ─────────────────────────────────────────────────────────
    float Ylm[15];
    int n_lm = 0;
    
    Ylm[n_lm++] = SH_C1_0 * y;
    Ylm[n_lm++] = SH_C1_1 * z;
    Ylm[n_lm++] = SH_C1_2 * x;
    
    if (sh_dim3 > 4)
        {
            xx=x*x; yy=y*y; zz=z*z; xy=x*y; yz=y*z; xz=x*z;
            Ylm[n_lm++] = SH_C2_0 * xy;
            Ylm[n_lm++] = SH_C2_1 * yz;
            Ylm[n_lm++] = SH_C2_2 * (2.f*zz - xx - yy);
            Ylm[n_lm++] = SH_C2_3 * xz;
            Ylm[n_lm++] = SH_C2_4 * (xx - yy);
            
            if (sh_dim3 > 9)
                {
                    Ylm[n_lm++] = SH_C3_0 * y * (3.f*xx - yy);
                    Ylm[n_lm++] = SH_C3_1 * xy * z;
                    Ylm[n_lm++] = SH_C3_2 * y * (4.f*zz - xx - yy);
                    Ylm[n_lm++] = SH_C3_3 * z * (2.f*zz - 3.f*xx - 3.f*yy);
                    Ylm[n_lm++] = SH_C3_4 * x * (4.f*zz - xx - yy);
                    Ylm[n_lm++] = SH_C3_5 * z * (xx - yy);
                    Ylm[n_lm++] = SH_C3_6 * x * (xx - 3.f*yy);
                }
        }
    // n_lm == L
    
    // ── Save Ylm for backward ─────────────────────────────────────────────────
    float *Ylm_out = Ylm_stored + i * L;
    for (int lm = 0; lm < n_lm; ++lm)
        Ylm_out[lm] = Ylm[lm];
    
    // ── Fast contraction (rank=1) ─────────────────────────────────────────────
    const int   P = (K * L) / Q;
    const float *A = shs_A + i * P;
    const float *B = shs_B + i * Q;
    
    for (int k = 0; k < K; ++k)
        {
            float acc = 0.f;
            for (int lm = 0; lm < L; ++lm)
                {
                    const int   g   = lm * K + k;
                    const int   row = g / Q;
                    const int   col = g % Q;
                    acc += A[row] * B[col] * Ylm[lm];
                }
            weight[k] += acc;
        }
    
    // ── Write weights ─────────────────────────────────────────────────────────
    float *w_out = weights + i * K;
    for (int k = 0; k < K; ++k)
        w_out[k] = weight[k];
}


// ─────────────────────────────────────────────────────────────────────────────
// sh2Weight_fast_B  (backward, rank=1)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void sh2Weight_fast_B(
    int32_t gs_num,
    const float *__restrict__ shs_A,       // (N, P)
    const float *__restrict__ shs_B,       // (N, Q)
    const float *__restrict__ Ylm_stored,  // (N, L)
    const float *__restrict__ dloss_dw,    // (N, K)
    const int K,
    const int L,
    const int Q,
    float *__restrict__ dloss_dA,          // (N, P)
    float *__restrict__ dloss_dB           // (N, Q)
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gs_num) return;
    
    const int P = K * L / Q;
    
    const float *A   = shs_A      + i * P;
    const float *B   = shs_B      + i * Q;
    const float *Ylm = Ylm_stored + i * L;
    const float *dw  = dloss_dw   + i * K;
    float       *dA  = dloss_dA   + i * P;
    float       *dB  = dloss_dB   + i * Q;
    
    for (int k = 0; k < K; ++k)
        {
            const float dw_k = dw[k];
            for (int lm = 0; lm < L; ++lm)
                {
                    const int   g   = lm * K + k;
                    const int   row = g / Q;
                    const int   col = g % Q;
                    const float ylm = Ylm[lm];
                    atomicAdd(&dA[row], dw_k * B[col] * ylm);
                    atomicAdd(&dB[col], dw_k * A[row] * ylm);
                }
        }
}



// ─────────────────────────────────────────────────────────────────────────────
// sh2Color
// ─────────────────────────────────────────────────────────────────────────────

__global__ void sh2Color(
    int32_t gs_num,
    const float3 *__restrict__ shs,
    const float3 *__restrict__ pws,
    const float3 *__restrict__ twc,
    const int sh_dim3,
    float3 *__restrict__ colors,
    float *__restrict__ dcolor_dshs,
    float *__restrict__ dcolor_dpws)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gs_num) return;

    float dc_dsh1,  dc_dsh2,  dc_dsh3;
    float dc_dsh4,  dc_dsh5,  dc_dsh6,  dc_dsh7,  dc_dsh8;
    float dc_dsh9,  dc_dsh10, dc_dsh11, dc_dsh12, dc_dsh13, dc_dsh14, dc_dsh15;
    float3 sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8;
    float3 sh9, sh10, sh11, sh12, sh13, sh14, sh15;
    float d0, d1, d2, normd_inv;
    float x, y, z, xx, yy, zz, xy, xz, yz;

    float3 sh0    = shs[i * sh_dim3 + 0];
    float3 color  = SH_C0_0 * sh0;

    if (sh_dim3 > 1)
    {
        float3 dv = pws[i] - twc[0];
        d0 = dv.x; d1 = dv.y; d2 = dv.z;
        normd_inv = 1.f / sqrt(d0*d0 + d1*d1 + d2*d2);
        float3 dir = dv * normd_inv;
        x = dir.x; y = dir.y; z = dir.z;

        sh1 = shs[i*sh_dim3+1];
        sh2 = shs[i*sh_dim3+2];
        sh3 = shs[i*sh_dim3+3];
        dc_dsh1 = SH_C1_0*y; dc_dsh2 = SH_C1_1*z; dc_dsh3 = SH_C1_2*x;
        color += dc_dsh1*sh1 + dc_dsh2*sh2 + dc_dsh3*sh3;

        if (sh_dim3 > 4)
        {
            xx=x*x; yy=y*y; zz=z*z; xy=x*y; yz=y*z; xz=x*z;
            sh4=shs[i*sh_dim3+4]; sh5=shs[i*sh_dim3+5]; sh6=shs[i*sh_dim3+6];
            sh7=shs[i*sh_dim3+7]; sh8=shs[i*sh_dim3+8];
            dc_dsh4=SH_C2_0*xy;
            dc_dsh5=SH_C2_1*yz;
            dc_dsh6=SH_C2_2*(2.f*zz-xx-yy);
            dc_dsh7=SH_C2_3*xz;
            dc_dsh8=SH_C2_4*(xx-yy);
            color += dc_dsh4*sh4 + dc_dsh5*sh5 + dc_dsh6*sh6
                   + dc_dsh7*sh7 + dc_dsh8*sh8;

            if (sh_dim3 > 9)
            {
                sh9 =shs[i*sh_dim3+ 9]; sh10=shs[i*sh_dim3+10];
                sh11=shs[i*sh_dim3+11]; sh12=shs[i*sh_dim3+12];
                sh13=shs[i*sh_dim3+13]; sh14=shs[i*sh_dim3+14];
                sh15=shs[i*sh_dim3+15];
                dc_dsh9  = SH_C3_0*y*(3.f*xx-yy);
                dc_dsh10 = SH_C3_1*xy*z;
                dc_dsh11 = SH_C3_2*y*(4.f*zz-xx-yy);
                dc_dsh12 = SH_C3_3*z*(2.f*zz-3.f*xx-3.f*yy);
                dc_dsh13 = SH_C3_4*x*(4.f*zz-xx-yy);
                dc_dsh14 = SH_C3_5*z*(xx-yy);
                dc_dsh15 = SH_C3_6*x*(xx-3.f*yy);
                color += dc_dsh9*sh9   + dc_dsh10*sh10 + dc_dsh11*sh11
                       + dc_dsh12*sh12 + dc_dsh13*sh13 + dc_dsh14*sh14
                       + dc_dsh15*sh15;
            }
        }
    }
    colors[i] = color;

    if (dcolor_dshs != nullptr && dcolor_dpws != nullptr)
    {
        float3 dc_dr0 = {0,0,0}, dc_dr1 = {0,0,0}, dc_dr2 = {0,0,0};
        Matrix<3,3> dr_dpw = {0,0,0, 0,0,0, 0,0,0};

        dcolor_dshs[sh_dim3*i+0] = SH_C0_0;

        if (sh_dim3 > 1)
        {
            const float normd3_inv = normd_inv*normd_inv*normd_inv;
            const float dr00=-d0*d0*normd3_inv+normd_inv, dr11=-d1*d1*normd3_inv+normd_inv;
            const float dr22=-d2*d2*normd3_inv+normd_inv, dr01=-d0*d1*normd3_inv;
            const float dr02=-d0*d2*normd3_inv,            dr12=-d1*d2*normd3_inv;
            dr_dpw = {dr00,dr01,dr02, dr01,dr11,dr12, dr02,dr12,dr22};

            dcolor_dshs[sh_dim3*i+1]=dc_dsh1;
            dcolor_dshs[sh_dim3*i+2]=dc_dsh2;
            dcolor_dshs[sh_dim3*i+3]=dc_dsh3;
            dc_dr0 += SH_C1_2*sh3;
            dc_dr1 += SH_C1_0*sh1;
            dc_dr2 += SH_C1_1*sh2;

            if (sh_dim3 > 4)
            {
                dcolor_dshs[sh_dim3*i+4]=dc_dsh4; dcolor_dshs[sh_dim3*i+5]=dc_dsh5;
                dcolor_dshs[sh_dim3*i+6]=dc_dsh6; dcolor_dshs[sh_dim3*i+7]=dc_dsh7;
                dcolor_dshs[sh_dim3*i+8]=dc_dsh8;
                dc_dr0 += SH_C2_0*y*sh4 - SH_C2_2*2*x*sh6 + SH_C2_3*z*sh7 + SH_C2_4*2*x*sh8;
                dc_dr1 += SH_C2_0*x*sh4 + SH_C2_1*z*sh5 - SH_C2_2*2.f*y*sh6 - SH_C2_4*2*y*sh8;
                dc_dr2 += SH_C2_1*y*sh5 + SH_C2_2*4.f*z*sh6 + SH_C2_3*x*sh7;

                if (sh_dim3 > 9)
                {
                    dcolor_dshs[sh_dim3*i+ 9]=dc_dsh9;  dcolor_dshs[sh_dim3*i+10]=dc_dsh10;
                    dcolor_dshs[sh_dim3*i+11]=dc_dsh11; dcolor_dshs[sh_dim3*i+12]=dc_dsh12;
                    dcolor_dshs[sh_dim3*i+13]=dc_dsh13; dcolor_dshs[sh_dim3*i+14]=dc_dsh14;
                    dcolor_dshs[sh_dim3*i+15]=dc_dsh15;
                    dc_dr0 += 6.f*SH_C3_0*sh9*x*y + SH_C3_1*sh10*yz - 2*SH_C3_2*sh11*xy
                            - 6.f*SH_C3_3*sh12*xz + SH_C3_4*sh13*(4.f*zz-3.f*xx-yy)
                            + 2*SH_C3_5*sh14*xz + SH_C3_6*sh15*(3*xx-3*yy);
                    dc_dr1 += SH_C3_0*sh9*(3.f*xx-2.f*yy) + SH_C3_1*sh10*xz
                            + SH_C3_2*sh11*(4.f*zz-xx-3.f*yy) - 6.f*SH_C3_3*sh12*yz
                            + SH_C3_4*sh13*(-2*xy) - 2*SH_C3_5*sh14*yz - 6.f*SH_C3_6*sh15*xy;
                    dc_dr2 += SH_C3_1*sh10*xy + 8.f*SH_C3_2*sh11*yz
                            + SH_C3_3*sh12*(6.f*zz-3.f*xx-3.f*yy)
                            + 8.f*SH_C3_4*sh13*xz + SH_C3_5*sh14*(xx-yy);
                }
            }
        }

        Matrix<3,3> dc_dr = {
            dc_dr0.x, dc_dr1.x, dc_dr2.x,
            dc_dr0.y, dc_dr1.y, dc_dr2.y,
            dc_dr0.z, dc_dr1.z, dc_dr2.z};
        Matrix<3,3> dcolor_dpw = dc_dr * dr_dpw;
        for (int j = 0; j < 9; j++) dcolor_dpws[i*9 + j] = dcolor_dpw(j);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Backward: color splatting
// ─────────────────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(BLOCK * BLOCK) draw_color_B(
    const int width,
    const int height,
    const int2 *__restrict__ patch_range_per_tile,
    const int *__restrict__ gsid_per_patch,
    const float2 *__restrict__ us,
    const float3 *__restrict__ cinv2ds,
    const float *__restrict__ alphas,
    const float3 *__restrict__ colors,
    const float *__restrict__ final_tau,
    const int *__restrict__ contrib,
    const float *__restrict__ dloss_dgammas,
    float2 *__restrict__ dloss_dus,
    float3 *__restrict__ dloss_dcinv2ds,
    float *__restrict__ dloss_dalphas,
    float *__restrict__ dloss_dcolors)
{
    const uint2 tile = {blockIdx.x, blockIdx.y};
    const uint2 pix  = {tile.x * BLOCK + threadIdx.x,
                        tile.y * BLOCK + threadIdx.y};

    const int      tile_idx = tile.y * gridDim.x + tile.x;
    const uint32_t pix_idx  = width * pix.y + pix.x;

    const bool inside = pix.x < width && pix.y < height;
    const int2 range  = patch_range_per_tile[tile_idx];
    const int  gs_num = range.y - range.x;
    if (gs_num == 0) return;

    bool thread_is_finished = !inside;

    __shared__ float2 shared_pos2d [BLOCK_SIZE];
    __shared__ float3 shared_cinv2d[BLOCK_SIZE];
    __shared__ float  shared_alpha [BLOCK_SIZE];
    __shared__ float3 shared_color [BLOCK_SIZE];
    __shared__ int    shared_gsid  [BLOCK_SIZE];

    float3 gamma_cur2last = {0.f, 0.f, 0.f};
    float3 dloss_dgamma   = {0.f, 0.f, 0.f};
    float  tau = 0.f;
    int    cont = 0;

    if (inside)
    {
        dloss_dgamma = {dloss_dgammas[0*height*width + pix_idx],
                        dloss_dgammas[1*height*width + pix_idx],
                        dloss_dgammas[2*height*width + pix_idx]};
        tau  = final_tau[pix_idx];
        cont = contrib[pix_idx];
    }

    for (int i = 0; i < gs_num; i++)
    {
        int finished = __syncthreads_count(thread_is_finished);
        if (finished == BLOCK_SIZE) break;

        int j = i % BLOCK_SIZE;
        if (j == 0)
        {
            fetch2shared_color(i / BLOCK_SIZE, true, range,
                               gsid_per_patch, us, cinv2ds, alphas, colors,
                               shared_pos2d, shared_cinv2d, shared_alpha,
                               shared_color, shared_gsid);
            __syncthreads();
        }

        if (i < gs_num - cont) continue;

        const float2 u      = shared_pos2d[j];
        const float3 cinv   = shared_cinv2d[j];
        const float  alpha  = shared_alpha[j];
        const float3 color  = shared_color[j];
        const int    gs_id  = shared_gsid[j];
        const float2 d      = u - pix;

        const float maha_dist   = max(0.f, mahaSqDist(cinv, d));
        const float g           = exp(-0.5f * maha_dist);
        const float alpha_prime = min(0.99f, alpha * g);
        if (alpha_prime < 0.002f) continue;

        tau /= (1.f - alpha_prime);

        const float3 dgamma_dalphaprime = tau * (color - gamma_cur2last);
        const float  dloss_dalphaprime  = dot(dloss_dgamma, dgamma_dalphaprime);
        atomicAdd(&dloss_dalphas[gs_id], dloss_dalphaprime * g);

        const float  dgamma_dcolor = alpha_prime * tau;
        float3 dloss_dcolor = dloss_dgamma * dgamma_dcolor;
        atomicAdd(&dloss_dcolors[gs_id*3+0], dloss_dcolor.x);
        atomicAdd(&dloss_dcolors[gs_id*3+1], dloss_dcolor.y);
        atomicAdd(&dloss_dcolors[gs_id*3+2], dloss_dcolor.z);

        float2 dalphaprime_du = {(-cinv.x*d.x - cinv.y*d.y)*alpha_prime,
                                 (-cinv.y*d.x - cinv.z*d.y)*alpha_prime};
        float2 dloss_du = dloss_dalphaprime * dalphaprime_du;
        atomicAdd(&dloss_dus[gs_id].x, dloss_du.x);
        atomicAdd(&dloss_dus[gs_id].y, dloss_du.y);

        float3 dalphaprime_dcinv = {-0.5f*alpha_prime*(d.x*d.x),
                                    -1.0f*alpha_prime*(d.x*d.y),
                                    -0.5f*alpha_prime*(d.y*d.y)};
        float3 dloss_dcinv = dloss_dalphaprime * dalphaprime_dcinv;
        atomicAdd(&dloss_dcinv2ds[gs_id].x, dloss_dcinv.x);
        atomicAdd(&dloss_dcinv2ds[gs_id].y, dloss_dcinv.y);
        atomicAdd(&dloss_dcinv2ds[gs_id].z, dloss_dcinv.z);

        gamma_cur2last = alpha_prime * color + (1.f - alpha_prime) * gamma_cur2last;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Backward: weight splatting
// ─────────────────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(BLOCK * BLOCK) draw_weight_B(
    const int width,
    const int height,
    const int2 *__restrict__ patch_range_per_tile,
    const int *__restrict__ gsid_per_patch,
    const float2 *__restrict__ us,
    const float3 *__restrict__ cinv2ds,
    const float *__restrict__ alphas,
    const float *__restrict__ weights,
    const float *__restrict__ final_tau,
    const int *__restrict__ contrib,
    const float *__restrict__ dloss_dgammas,
    float *__restrict__ dloss_dweights,
    const int num_palette)
{
    int splat_dim = num_palette;
    
    const uint2 tile = {blockIdx.x, blockIdx.y};
    const uint2 pix  = {tile.x * BLOCK + threadIdx.x,
        tile.y * BLOCK + threadIdx.y};
    
    const int      tile_idx = tile.y * gridDim.x + tile.x;
    const uint32_t pix_idx  = width * pix.y + pix.x;
    
    const bool inside = pix.x < width && pix.y < height;
    const int2 range  = patch_range_per_tile[tile_idx];
    const int  gs_num = range.y - range.x;
    if (gs_num == 0) return;
    
    bool thread_is_finished = !inside;
    
    __shared__ float2 shared_pos2d [BLOCK_SIZE];
    __shared__ float3 shared_cinv2d[BLOCK_SIZE];
    __shared__ float  shared_alpha [BLOCK_SIZE];
    __shared__ float  shared_weight[BLOCK_SIZE * MAX_SPLAT_DIM];
    __shared__ int    shared_gsid  [BLOCK_SIZE];
    
    float dloss_dgamma[MAX_SPLAT_DIM];
    for (int k = 0; k < splat_dim; ++k) dloss_dgamma[k] = 0.f;
    
    float tau  = 0.f;
    int   cont = 0;
    
    if (inside)
        {
            for (int k = 0; k < splat_dim; ++k)
                dloss_dgamma[k] = dloss_dgammas[k * height * width + pix_idx];
            tau  = final_tau[pix_idx];
            cont = contrib[pix_idx];
        }
    
    for (int i = 0; i < gs_num; i++)
        {
            int finished = __syncthreads_count(thread_is_finished);
            if (finished == BLOCK_SIZE) break;
            
            int j = i % BLOCK_SIZE;
            if (j == 0)
                {
                    fetch2shared_weight(i / BLOCK_SIZE, true, range,
                        gsid_per_patch, us, cinv2ds, alphas, weights,
                        shared_pos2d, shared_cinv2d, shared_alpha,
                        shared_weight, shared_gsid, num_palette);
                    __syncthreads();
                }
            
            if (i < gs_num - cont) continue;
            
            const float2 u     = shared_pos2d[j];
            const float3 cinv  = shared_cinv2d[j];
            const float  alpha = shared_alpha[j];
            float       *w     = &shared_weight[j * splat_dim];
            const int    gs_id = shared_gsid[j];
            const float2 d     = u - pix;
            
            const float maha_dist   = max(0.f, mahaSqDist(cinv, d));
            const float g           = exp(-0.5f * maha_dist);
            const float alpha_prime = min(0.99f, alpha * g);
            if (alpha_prime < 0.002f) continue;
            
            tau /= (1.f - alpha_prime);
            
            const float dgamma_dweight = alpha_prime * tau;
            #pragma unroll
            for (int k = 0; k < splat_dim; ++k)
                atomicAdd(&dloss_dweights[gs_id * splat_dim + k],
                    dloss_dgamma[k] * dgamma_dweight);
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lab to RGB conversion (GPU)
// ─────────────────────────────────────────────────────────────────────────────

inline __device__ float lab_f_inv(float t)
{
    const float delta = 6.0f / 29.0f;
    return (t > delta) ? t*t*t : 3.f*delta*delta*(t - 4.f/29.f);
}

inline __device__ float xyz_to_rgb_component(float c)
{
    return (c <= 0.0031308f) ? 12.92f*c : 1.055f*powf(c, 1.f/2.4f) - 0.055f;
}

__global__ void lab2rgb(
    const int height,
    const int width,
    const float *__restrict__ lab_image,
    float *__restrict__ rgb_image)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    const int pix_idx    = py * width + px;
    const int total_pix  = height * width;

    const float L = lab_image[0*total_pix + pix_idx] * 100.f;
    const float a = lab_image[1*total_pix + pix_idx];
    const float b = lab_image[2*total_pix + pix_idx];

    const float fy = (L + 16.f) / 116.f;
    const float fx = a / 500.f + fy;
    const float fz = fy - b / 200.f;

    const float X = 0.95047f * lab_f_inv(fx);
    const float Y = 1.00000f * lab_f_inv(fy);
    const float Z = 1.08883f * lab_f_inv(fz);

    float r = fmaxf(0.f, fminf(1.f,  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z));
    float g = fmaxf(0.f, fminf(1.f, -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z));
    float bv= fmaxf(0.f, fminf(1.f,  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z));

    rgb_image[0*total_pix + pix_idx] = xyz_to_rgb_component(r);
    rgb_image[1*total_pix + pix_idx] = xyz_to_rgb_component(g);
    rgb_image[2*total_pix + pix_idx] = xyz_to_rgb_component(bv);
}
