#include "/usr/include/stdio.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex> // std::mutex, std::unique_lock
#include <cmath>
#include <stdlib.h>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define OPENCV

#include "yolo_v2_class.hpp" // imported functions from DLL

#ifdef OPENCV
#ifdef ZED_STEREO
#include <sl/Camera.hpp>
#if ZED_SDK_MAJOR_VERSION == 2
#define ZED_STEREO_2_COMPAT_MODE
#endif

#undef GPU // avoid conflict with sl::MEM::GPU

#ifdef ZED_STEREO_2_COMPAT_MODE
#pragma comment(lib, "sl_core64.lib")
#pragma comment(lib, "sl_input64.lib")
#endif
#pragma comment(lib, "sl_zed64.lib")

float getMedian(std::vector<float> &v) {
size_t n = v.size() / 2;
std::nth_element(v.begin(), v.begin() + n, v.end());
return v[n];
}

std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba)
{
bool valid_measure;
int i, j;
const unsigned int R_max_global = 10;

std::vector<bbox_t> bbox3d_vect;

for (auto &cur_box : bbox_vect) {

const unsigned int obj_size = std::min(cur_box.w, cur_box.h);
const unsigned int R_max = std::min(R_max_global, obj_size / 2);
int center_i = cur_box.x + cur_box.w * 0.5f, center_j = cur_box.y + cur_box.h * 0.5f;

std::vector<float> x_vect, y_vect, z_vect;
for (int R = 0; R < R_max; R++) {
for (int y = -R; y <= R; y++) {
for (int x = -R; x <= R; x++) {
i = center_i + x;
j = center_j + y;
sl::float4 out(NAN, NAN, NAN, NAN);
if (i >= 0 && i < xyzrgba.cols && j >= 0 && j < xyzrgba.rows) {
cv::Vec4f &elem = xyzrgba.at<cv::Vec4f>(j, i); // x,y,z,w
out.x = elem[0];
out.y = elem[1];
out.z = elem[2];
out.w = elem[3];
}
valid_measure = std::isfinite(out.z);
if (valid_measure)
{
x_vect.push_back(out.x);
y_vect.push_back(out.y);
z_vect.push_back(out.z);
}
}
}
}

if (x_vect.size() * y_vect.size() * z_vect.size() > 0)
{
cur_box.x_3d = getMedian(x_vect);
cur_box.y_3d = getMedian(y_vect);
cur_box.z_3d = getMedian(z_vect);
}
else {
cur_box.x_3d = NAN;
cur_box.y_3d = NAN;
cur_box.z_3d = NAN;
}

bbox3d_vect.emplace_back(cur_box);
}

return bbox3d_vect;
}

cv::Mat slMat2cvMat(sl::Mat &input) {
int cv_type = -1; // Mapping between MAT_TYPE and CV_TYPE
if(input.getDataType() ==
#ifdef ZED_STEREO_2_COMPAT_MODE
sl::MAT_TYPE_32F_C4
#else
sl::MAT_TYPE::F32_C4
#endif
) {
cv_type = CV_32FC4;
} else cv_type = CV_8UC4; // sl::Mat used are either RGBA images or XYZ (4C) point clouds
return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(
#ifdef ZED_STEREO_2_COMPAT_MODE
sl::MEM::MEM_CPU
#else
sl::MEM::CPU
#endif
));
}

cv::Mat zed_capture_rgb(sl::Camera &zed) {
sl::Mat left;
zed.retrieveImage(left);
cv::Mat left_rgb;
cv::cvtColor(slMat2cvMat(left), left_rgb, CV_RGBA2RGB);
return left_rgb;
}

cv::Mat zed_capture_3d(sl::Camera &zed) {
sl::Mat cur_cloud;
zed.retrieveMeasure(cur_cloud,
#ifdef ZED_STEREO_2_COMPAT_MODE
sl::MEASURE_XYZ
#else
sl::MEASURE::XYZ
#endif
);
return slMat2cvMat(cur_cloud).clone();
}

static sl::Camera zed; // ZED-camera

#else // ZED_STEREO
std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba) {
    return bbox_vect;
}
#endif // ZED_STEREO


#include <opencv2/opencv.hpp> // C++
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH // OpenCV 3.x and 4.x
#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#ifdef TRACK_OPTFLOW
/*
#pragma comment(lib, "opencv_cudaoptflow" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_cudaimgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
*/
#endif // TRACK_OPTFLOW
#endif // USE_CMAKE_LIBS
#else // OpenCV 2.x
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_video" OPENCV_VERSION ".lib")
#endif // USE_CMAKE_LIBS
#endif // CV_VERSION_EPOCH


void draw_boxes(cv::Mat mat_frame, std::vector<bbox_t> result_vec_face, std::vector<bbox_t> result_vec_phone,
                int current_det_fps = -1, int current_cap_fps = -1) {
    int const colors[6][3] = {{1, 0, 1},
                              {0, 0, 1},
                              {0, 1, 1},
                              {0, 1, 0},
                              {1, 1, 0},
                              {1, 0, 0}};


    for (int p = 0; p < result_vec_face.size(); ++p) {
        cv::Scalar color = obj_id_to_color(result_vec_face[p].obj_id);

        for (int i = 0; i < result_vec_phone.size(); ++i) {
            cv::Scalar color = obj_id_to_color(result_vec_phone[i].obj_id);

            cv::rectangle(mat_frame, cv::Rect(result_vec_face[p].x, result_vec_face[p].y, result_vec_face[p].w,
                                              result_vec_face[p].h), color, 2);

//printf("얼굴의 x좌표 %d 얼굴의 y좌표 %d\n", (result_vec_face[p].x - result_vec_face[p].w /2),
//(result_vec_face[p].y - result_vec_face[p].h) / 2);

            if (result_vec_phone[i].obj_id == 67) {

                cv::rectangle(mat_frame, cv::Rect(result_vec_phone[i].x, result_vec_phone[i].y, result_vec_phone[i].w,
                                                  result_vec_phone[i].h), color, 2);

//printf("휴대폰의 x좌표 %d 휴대폰의 y좌표 %d\n", (result_vec_phone[i].x - result_vec_phone[i].w) / 2,
//(result_vec_phone[i].y - result_vec_phone[i].h) / 2);


                double result1 = 0;
                double result2 = 0;
                double result3 = 0;
                result1 = sqrt(pow(abs((int) (result_vec_face[p].x + result_vec_face[p].w / 2 - result_vec_phone[i].x +
                                              result_vec_phone[i].w / 2)), 2) +
                               pow(abs((int) (result_vec_face[p].y + result_vec_face[p].h / 2 - result_vec_phone[i].y +
                                              result_vec_phone[i].h / 2)), 2));

                result2 = (abs((int) result_vec_face[p].x));
                result3 = (abs((int) (result_vec_face[p].w * result_vec_face[p].h)));
                result3 = (std::round(result3 * 1000)) / 1000.0;
                printf("\n박스넓이: %.2f", result3);
                printf("\nx좌표값: %.2f", result2);
                printf("\n거리출력: %.2f\n", result1);

                if (((abs((int) result3)) < 50000) && ((abs((int) result3)) > 10000)) {
                    printf("상태: 정상적인 거리에 있습니다");
                    if (result1 < 105) {
                        printf("\n################전화 사용감지################");
                    } else {
                        printf("\n################경고 휴대전화 인식################");
                    }
                } else if (((abs((int) result3)) < 100000) && ((abs((int) result3)) > 50000)) {
                    printf("상태: 가까운 거리에 있습니다!");
                    if (result1 < 400) {
                        printf("\n################전화 사용감지################");
                    } else {
                        printf("\n################경고 휴대전화 인식################");
                    }
                } else if (((abs((int) result3)) < 10000) && ((abs((int) result3)) > 10)) {
                    printf("상태: 먼 거리에 있습니다");
                    if (result1 < 100) {
                        printf("\n################전화 사용감지################");
                    } else {
                        printf("\n################경고 휴대전화 인식################");
                    }
                }
            }
        }
    }
}






#endif // OPENCV


void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ", x = " << i.x << ", y = " << i.y
                  << ", w = " << i.w << ", h = " << i.h
                  << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

template<typename T>
class send_one_replaceable_object_t {
    const bool sync;
    std::atomic<T *> a_ptr;
public:

    void send(T const& _obj) {
        T *new_ptr = new T;
        *new_ptr = _obj;
        if (sync) {
            while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
    }

    T receive() {
        std::unique_ptr<T> ptr;
        do {
            while(!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ptr.reset(a_ptr.exchange(NULL));
        } while (!ptr);
        T obj = *ptr;
        return obj;
    }

    bool is_object_present() {
        return (a_ptr.load() != NULL);
    }

    send_one_replaceable_object_t(bool _sync) : sync(_sync), a_ptr(NULL)
    {}
};



int main(int argc, char *argv[])
{

    std::string cfg_file_face = "/home/ycs/Desktop/darknet/yolo-face.cfg";
    std::string weights_file_face = "/home/ycs/Desktop/darknet/yolo-face_final.weights";

    std::string cfg_file_phone = "/home/ycs/Desktop/darknet/yolov4.cfg";
    std::string weights_file_phone = "/home/ycs/Desktop/darknet/yolov4.weights";
    Detector detector_face(cfg_file_face, weights_file_face);
    Detector detector_phone(cfg_file_phone, weights_file_phone);

    VideoCapture cap(0);
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    while(1) {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        cv::Mat mat_frame = frame;
        std::shared_ptr<image_t> det_image_face = detector_face.mat_to_image_resize(mat_frame);
        std::shared_ptr<image_t> det_image_phone = detector_phone.mat_to_image_resize(mat_frame);

        std::vector<bbox_t> result_vec_face = detector_face.detect_resized(*det_image_face, mat_frame.size().width,mat_frame.size().height);
        std::vector<bbox_t> result_vec_phone = detector_phone.detect_resized(*det_image_phone, mat_frame.size().width,mat_frame.size().height);

        draw_boxes(mat_frame, result_vec_face, result_vec_phone);
        imshow("Frame", frame);
        char c = (char) waitKey(25);
        if (c == 27)
            break;

    }
    cap.release();
    destroyAllWindows();

    return 0;

}