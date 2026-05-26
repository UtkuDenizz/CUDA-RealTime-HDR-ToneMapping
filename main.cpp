#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include "filter_utils.h"

// external call to our CUDA kernel wrapper
extern "C" void apply_hdr(unsigned char* d_input, unsigned char* d_output, int w, int h, float exp, float gam);

int main() {
    // attempt to open the default camera
    cv::VideoCapture cap(0);
    int width = 1280, height = 720; // default fallback resolution
    bool is_camera = cap.isOpened();

    if (is_camera) {
        width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    }

    // allocate gpu memory
    size_t size = width * height * 3;
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cv::Mat frame(height, width, CV_8UC3);
    cv::Mat output_frame(height, width, CV_8UC3);

    std::cout << (is_camera ? "Status: Camera Active" : "Status: Synthetic Mode (OpenCV Generator)") << std::endl;

    while (true) {
        if (is_camera) {
            cap >> frame;
            if (frame.empty()) break;
        } else {
            // generate synthetic data using OpenCV
            frame = cv::Scalar(25, 25, 25); // dark gray background
            static int pos = 0;
            // draw a moving green rectangle to simulate video
            cv::rectangle(frame, cv::Rect(pos % width, height / 3, 160, 160), cv::Scalar(0, 255, 0), -1);
            cv::putText(frame, "SYNTHETIC HDR SOURCE", cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
            pos += 15;
        }

        // copy frame to device, execute hdr kernel, and copy back to host
        cudaMemcpy(d_in, frame.data, size, cudaMemcpyHostToDevice);
        apply_hdr(d_in, d_out, width, height, 1.3f, 2.2f); // exposure=1.3, gamma=2.2
        cudaMemcpy(output_frame.data, d_out, size, cudaMemcpyDeviceToHost);

        // display results
        cv::imshow("CUDA HDR Tone Mapping - Press 'q' to exit", output_frame);
        if (cv::waitKey(1) == 'q') break;
    }

    // Cleanup
    cudaFree(d_in); 
    cudaFree(d_out);
    return 0;
}