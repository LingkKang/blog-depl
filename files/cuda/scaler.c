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
 * Grayscaler with fast integer (BT.601).
 */
static inline unsigned char grayscale(unsigned char r, unsigned char g, unsigned char b)
{
    unsigned int y = 77u * r + 150u * g + 29u * b + 128u; /* +128 for rounding */
    return (unsigned char)(y >> 8);
}

int main(void)
{
    FILE *fpi = fopen("source.bmp", "rb");
    FILE *fpo = fopen("output.bmp", "wb");

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
        fprintf(stderr, "Unsupported BMP: expecting uncompressed 24 BPP.\n");
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
    unsigned char *img = (unsigned char *)malloc(img_size);

    double tA = now_sec();
    (void)!fread(img, 1, img_size, fpi);
    double tB = now_sec();
    double t_io_read = tB - tA;

    double t_compute = 0.0;
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            double tC = now_sec();
            int32_t i = (y * width + x);
            unsigned char *p = &img[i * 3]; // One pixel, BGR.
            unsigned char gray = grayscale(p[2], p[1], p[0]);
            p[0] = gray;
            p[1] = gray;
            p[2] = gray;
            double tD = now_sec();
            t_compute += (tD - tC);
        }
    }

    double tE = now_sec();

    fwrite(img, 1, img_size, fpo);
    double tF = now_sec();
    double t_io_write = tF - tE;

    free(img);
    fclose(fpi);
    fclose(fpo);

    double t1 = now_sec();
    double t_total = t1 - t0;

    fprintf(stderr,
            "Total: %.4f s | Read: %.4f s | Compute: %.4f s | Write: %.4f s\n",
            t_total, t_io_read, t_compute, t_io_write);

    return 0;
}
