#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

const size_t HEADER_STR = 0x00; // Starting byte of BITMAPFILEHEADER.
const size_t HEADER_END = 0x0D;
const size_t HEADER_SIZE = HEADER_END - HEADER_STR + 1;
const size_t COMMON_INFO_SIZE = 40; // Common part of BITMAPINFOHEADER.

/*
 * Little-endian 32-bit integer reader.
 */
static inline uint32_t le32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/*
 * Little-endian 16-bit integer reader.
 */
static inline uint16_t le16(const unsigned char *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/*
 * The CUDA kernel for pixel grayscaling. Each thread handles a specific pixel.
 */
__global__ void grayscaleKernel(unsigned char *__restrict__ img, int pix_num)
{
    int pix = (blockDim.x * blockIdx.x + threadIdx.x);
    if (pix >= pix_num)
        return;

    int i = pix * 3;
    unsigned int y = 77u * img[i + 2] + 150u * img[i + 1] + 29u * img[i];
    unsigned char gray = (unsigned char)(y >> 8);
    img[i] = gray;
    img[i + 1] = gray;
    img[i + 2] = gray;
}

int main(void)
{
    FILE *fpi = fopen("source.bmp", "rb");
    FILE *fpo = fopen("output_cuda.bmp", "wb");

    if (!fpi || !fpo)
        return 1;

    double t0 = now_sec();

    // Load header and common info header.
    unsigned char header[HEADER_SIZE];
    unsigned char common_info[COMMON_INFO_SIZE];

    (void)!fread(header, 1, HEADER_SIZE, fpi);
    (void)!fread(common_info, 1, COMMON_INFO_SIZE, fpi);
    uint32_t data_offset = le32(&header[10]);
    uint32_t remain_info_size = data_offset - HEADER_SIZE - COMMON_INFO_SIZE;
    uint32_t width = le32(&common_info[4]);
    uint32_t height = le32(&common_info[8]);
    uint16_t bpp = le16(&common_info[14]);         // Bits per pixel.
    uint16_t compression = le32(&common_info[16]); // Compression.

    if (bpp != 24 || compression != 0)
    {
        fprintf(stderr, "Unsupported BMP: expecting 24bbp, uncompressed.\n");
        return 1;
    }

    // Load remaining info header.
    unsigned char remain_info[remain_info_size];
    (void)!fread(remain_info, 1, remain_info_size, fpi);

    // Write header and info header to output image.
    fwrite(header, 1, HEADER_SIZE, fpo);
    fwrite(common_info, 1, COMMON_INFO_SIZE, fpo);
    fwrite(remain_info, 1, remain_info_size, fpo);

    uint32_t pix_num = width * height;
    uint32_t img_size = pix_num * 3u;
    unsigned char *img_h = (unsigned char *)malloc(img_size);

    double tA = now_sec();
    (void)!fread(img_h, 1, img_size, fpi);
    double tB = now_sec();
    double t_io_read = tB - tA;

    unsigned char *img_d;

    cudaMalloc((void **)&img_d, img_size);
    double tC = now_sec();
    cudaMemcpy(img_d, img_h, img_size, cudaMemcpyHostToDevice);
    double tD = now_sec();
    double t_memcpy_h2d = tD - tC;

    int threads_per_block = (width > 1024) ? 1024 : width;
    int blocks_per_grid = (pix_num + threads_per_block - 1) / threads_per_block;
    double tE = now_sec();
    grayscaleKernel<<<blocks_per_grid, threads_per_block>>>(img_d, pix_num);
    cudaDeviceSynchronize();
    double tF = now_sec();
    double t_compute = tF - tE;

    double tG = now_sec();
    cudaMemcpy(img_h, img_d, img_size, cudaMemcpyDeviceToHost);
    double tH = now_sec();
    double t_memcpy_d2h = tH - tG;

    cudaFree(img_d);

    double tI = now_sec();
    fwrite(img_h, 1, img_size, fpo);
    double tJ = now_sec();
    double t_io_write = tJ - tI;

    free(img_h);
    fclose(fpi);
    fclose(fpo);

    double t1 = now_sec();
    double t_total = t1 - t0;

    fprintf(stderr,
            "Total: %.4f s | Read: %.4f s | H2D: %.4f s | Compute: %.4f s | D2H: %.4f s | Write: %.4f s\n",
            t_total, t_io_read, t_memcpy_h2d, t_compute, t_memcpy_d2h, t_io_write);

    return 0;
}
