#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <chrono>

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#endif

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS

#include "CL/opencl.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.hpp"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef ANALYZE
#define ANALYZE 1
#endif

#define dbgs                                                                   \
  if (!ANALYZE) {                                                              \
  } else                                                                       \
    std::cout

constexpr size_t ARR_SIZE = 2048;
constexpr size_t LOCAL_SIZE = 256;

#define STRINGIFY(...) #__VA_ARGS__

// This example have built-in kernel to easy modify, etc
// ---------------------------------- OpenCL ---------------------------------
const char* vakernel = STRINGIFY(
__kernel void img(__global uchar4 * image, int w, int h)
{
    int i = get_global_id(0);

    image[i].x *= 2;
    image[i].y /= 2;
    



    //float strength = 0.08f;
    //float speed = 0.08f;
    //int i = get_global_id(0);
    //int x = i % w;
    //int y = i / w;

    //// ÷ентр изображени€
    //float cx = w / 2.0f;
    //float cy = h / 2.0f;

    //// ѕолучаем угол дл€ каждого пиксел€ относительно центра
    //float dx = (float)(x - cx);
    //float dy = (float)(y - cy);
    //float distance = sqrt(dx * dx + dy * dy);

    //// ¬ычисл€ем угол дл€ текущего пиксел€
    //float angle = atan2(dy, dx);

    //// »змен€ем угол на основе рассто€ни€ и времени
    //angle += strength * sin(distance * speed);

    //// ѕересчитываем новые координаты после вращени€
    //float newX = cx + cos(angle) * distance;
    //float newY = cy + sin(angle) * distance;

    //// ѕытаемс€ отобразить пиксель в новых координатах
    //int nx = (int)newX;
    //int ny = (int)newY;

    //// ≈сли новые координаты выход€т за пределы, оставл€ем их в пределах изображени€
    //if (nx < 0) nx = 0;
    //if (ny < 0) ny = 0;
    //if (nx >= w) nx = w - 1;
    //if (ny >= h) ny = h - 1;

    //// ѕолучаем исходный пиксель
    //uchar4 pixel = image[ny * w + nx];

    //// «аписываем пиксель в новое место
    //image[y * w + x] = pixel;




}


);
// ---------------------------------- OpenCL ---------------------------------


struct RGBA {
    uint8_t r, g, b, a;
};

// OpenCL application encapsulates platform, context and queue
// We can offload vector addition through its public interface
class OclApp {
    cl::Platform P_;
    cl::Context C_;
    cl::CommandQueue Q_;

    static cl::Platform select_platform();
    static cl::Context get_gpu_context(cl_platform_id);
    
    using vadd_t = cl::KernelFunctor<cl::Buffer, int, cl::Buffer>;
    using img_t = cl::KernelFunctor<cl::Buffer, int, int>;

public:
    OclApp() : P_(select_platform()), C_(get_gpu_context(P_())), Q_(C_) {
        cl::string name = P_.getInfo<CL_PLATFORM_NAME>();
        cl::string profile = P_.getInfo<CL_PLATFORM_PROFILE>();
        dbgs << "Selected: " << name << ": " << profile << std::endl;
    }


    void vadd(cl_int const* A, cl_int n, cl_int* C);
    void test_img(RGBA* img, int w, int h);
};



std::vector<RGBA> LoadImage(const std::string& filename, int& width, int& height) {
    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 4);

    if (!data) {
        throw std::runtime_error("Load image failure [FILE NAME]: " + filename);
    }

    std::vector<RGBA> image(width * height);
    std::memcpy(image.data(), data, width * height * 4);

    stbi_image_free(data);
    return image;
}



void SaveImage(const std::string& filename, const std::vector<RGBA>& image, int width, int height) {
    if (image.empty()) {
        throw std::runtime_error("Image empty: " + filename);
    }

    if (!stbi_write_png(filename.c_str(), width, height, 4, image.data(), width * 4)) {
        throw std::runtime_error("Fail to save image: " + filename);
    }

    std::cout << "Image save as: " << filename << std::endl;
}

// select first platform with some GPUs
cl::Platform OclApp::select_platform() {
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto p : platforms) {
        // note: usage of p() for plain id
        cl_uint numdevices = 0;
        ::clGetDeviceIDs(p(), CL_DEVICE_TYPE_GPU, 0, NULL, &numdevices);
        if (numdevices > 0)
            return cl::Platform(p); // retain?
    }
    throw std::runtime_error("No platform selected");
}

// get context for selected platform
cl::Context OclApp::get_gpu_context(cl_platform_id PId) {
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(PId),
        0 // signals end of property list
    };

    return cl::Context(CL_DEVICE_TYPE_GPU, properties);
}

void OclApp::vadd(cl_int const* APtr, cl_int n, cl_int* CPtr) {
    size_t BufSz = n * sizeof(cl_int);
   

    cl::Buffer A(C_, CL_MEM_READ_ONLY, BufSz);
    cl::Buffer C(C_, CL_MEM_WRITE_ONLY, sizeof(cl_int));




    cl::copy(Q_, APtr, APtr + n, A);
    cl::copy(Q_, CPtr, CPtr+1, C);

    // try forget context here and happy debugging CL_INVALID_MEM_OBJECT:
    // cl::Program program(vakernel, true /* build immediately */);
    cl::Program program(C_, vakernel, true /* build immediately */);

    vadd_t add_vecs(program, "vector_add");

    cl::NDRange GlobalRange(n);
    cl::NDRange LocalRange(LOCAL_SIZE);
    cl::EnqueueArgs Args(Q_, GlobalRange, LocalRange);

    cl::Event evt = add_vecs(Args, A, n, C);
    evt.wait();
    

    cl::copy(Q_, C, CPtr, CPtr+1);
}

void OclApp::test_img(RGBA* img, int w, int h)
{
    size_t NumPixels = w * h;
    size_t BufSz = NumPixels * sizeof(RGBA);

    cl::Buffer A(C_, CL_MEM_READ_WRITE, BufSz);

    cl::copy(Q_, img, img + NumPixels, A);


    cl::Program program(C_, vakernel, true);

    img_t img_tst(program, "img");

    cl::NDRange GlobalRange(NumPixels);
    cl::NDRange LocalRange(LOCAL_SIZE);


    cl::EnqueueArgs Args(Q_, GlobalRange, LocalRange);
    cl::Event evt = img_tst(Args, A, w, h);
    evt.wait();




    cl::copy(Q_, A, img, img + NumPixels);
}

int main() try {
    int w;
    int h;



    OclApp app;
    cl::vector<RGBA> src = LoadImage("../res/giraf.png", w, h);



    for (int i = 0; i < 1; i++) 
    {
        auto start = std::chrono::high_resolution_clock::now();
        app.test_img(src.data(), w, h);
        std::cout << i << "\t";
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << duration.count() << std::endl;
    }

#if 0
    for (int i = 0; i < src.size(); i++) {
        char gray = (char)(0.299f * src[i].r + 0.587f * src[i].g + 0.114f * src[i].b);

        src[i].r = gray;
        src[i].g = gray;
        src[i].b = gray;
    }
#endif
    

    SaveImage("../out.png", src, w, h);



    



    //app.vadd(src.data(), ARR_SIZE, dst.data());


    //std::cout << dst[0] << std::endl;

    


}
catch (cl::Error& err) {
    std::cerr << "OCL ERROR " << err.err() << ":" << err.what() << std::endl;
    return -1;
}
catch (std::runtime_error& err) {
    std::cerr << "RUNTIME ERROR " << err.what() << std::endl;
    return -1;
}
catch (...) {
    std::cerr << "UNKNOWN ERROR\n";
    return -1;
}