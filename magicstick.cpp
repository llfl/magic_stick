
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

// #include <caffe/caffe.hpp>

// Command-line user interface
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

#include <cmath>

using namespace std;
#define STICK_RELATIVE_LENGTH 3
#define DURATION 3
#define EXCLUTION 20
#define MIN_VELOCITY 100

double eps = 0.01;

int stallState = 0;

int poseState = -1;

vector<cv::Vec2f > stick_point;

//kalman filter

namespace kf {
    class kalmanFilter{
        public:
            kalmanFilter(int x, int y):KF_(6,4){
                measurement_ = cv::Mat::zeros(4,1,CV_32F);
                KF_.transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, (5./6.), 0, 0, 0,
                                                             0, 1, 0, (5./6.), 0, 0,
                                                             0, 0, 1, 0, 1, 0,
                                                             0, 0, 0, 1, 0, 1,
                                                             0, 0, 0, 0, 1, 0,
                                                             0, 0, 0, 0, 0, 1);
                setIdentity(KF_.measurementMatrix, cv::Scalar::all(1));
                setIdentity(KF_.processNoiseCov, cv::Scalar::all(1e-10));//**10: Larger, slower regression
                setIdentity(KF_.measurementNoiseCov, cv::Scalar::all(1));//1: Larger, quicker regression
                setIdentity(KF_.errorCovPost, cv::Scalar::all(1));
 
                KF_.statePost = (cv::Mat_<float>(6, 1) << x, y, 0, 0, 0, 0);
            }

            cv::Point2f run(float x1, float y1,float x2, float y2){
                cv::Mat prediction = KF_.predict();
                cv::Point2f predict_pt = cv::Point2f(prediction.at<float>(0),prediction.at<float>(1));

                measurement_.at<float>(0, 0) = x1;
                measurement_.at<float>(1, 0) = y1;
                measurement_.at<float>(2, 0) = x2;
                measurement_.at<float>(3, 0) = y2;

                KF_.correct(measurement_);

                return predict_pt;
            }

        private:
            cv::Mat measurement_;
            cv::KalmanFilter KF_;
    };

    class kalmanFilterW{
        public:
            kalmanFilterW(int x, int y):KF_(4,2){
                measurement_ = cv::Mat::zeros(2,1,CV_32F);
                KF_.transitionMatrix = (cv::Mat_<float>(4, 4) <<1, 0, 200, 0,
                                                                0, 1, 0, 200,
                                                                0, 0, 1, 0,
                                                                0, 0, 0, 1);
                setIdentity(KF_.measurementMatrix, cv::Scalar::all(1));
                setIdentity(KF_.processNoiseCov, cv::Scalar::all(1e-10));//**10: Larger, slower regression
                setIdentity(KF_.measurementNoiseCov, cv::Scalar::all(10));//1: Larger, quicker regression
                setIdentity(KF_.errorCovPost, cv::Scalar::all(10));
 
                KF_.statePost = (cv::Mat_<float>(4, 1) << x, y, 0, 0);
            }

            cv::Point2f run(float x, float y){
                cv::Mat prediction = KF_.predict();
                cv::Point2f predict_pt = cv::Point2f(prediction.at<float>(0),prediction.at<float>(1));

                measurement_.at<float>(0, 0) = x;
                measurement_.at<float>(1, 0) = y;

                KF_.correct(measurement_);

                return predict_pt;
            }

        private:
            cv::Mat measurement_;
            cv::KalmanFilter KF_;
    };
}

kf::kalmanFilterW kalman(0,0);

// This worker will just invert the image
class WUserPostProcessing : public op::Worker<std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>>
{
public:
    WUserPostProcessing()
    {
        // User's constructor here
    }

    void initializationOnThread() {}

    void work(std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
    {
        try
        {
            if (datumsPtr != nullptr && !datumsPtr->empty())
            {

                for (auto& datumPtr : *datumsPtr)
                {
                    cv::Mat cvOutputData = OP_OP2CVMAT(datumPtr->cvOutputData);
                    const auto& poseKeypoints = datumPtr->poseKeypoints;
                    for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++){

                        double RElbowx = poseKeypoints[{person, 3, 0}];
                        double RElbowy = poseKeypoints[{person, 3, 1}];
                        double RWristx = poseKeypoints[{person, 4, 0}];
                        double RWristy = poseKeypoints[{person, 4, 1}];
                        if(RElbowx<eps || RElbowy < eps || RWristx < eps || RWristy < eps){
                            stick_point.clear();
                            continue;
                        }
                        cv::Point2f stick_end = cv::Point2f(RWristx, RWristy);
                        double gradientK = (RWristy - RElbowy)/(RWristx - RElbowx);

                        if(stick_point.size() == 0){
                            stick_point.push_back(stick_end);
                            continue;
                        }
                        cv::Point2f stick_last = stick_point.back();
                        double gradientG = (stick_end.y - stick_last.y)/(stick_end.x - stick_last.x);
                        switch(poseState){
                            case 1:
                                    if(gradientG < -0.5 && gradientG >= -2 && stick_end.x < stick_last.x){
                                        poseState = 2;
                                    }else if( (gradientG < -2 || gradientG > 2) && stick_end.y > stick_last.y){
                                        poseState = 31;
                                    }else if(gradientG >= -0.5 && gradientG < 0.5 && stick_end.x < stick_last.x){
                                        poseState = 32;
                                    }else{
                                        poseState = 4;
                                    }
                                    break;
                            case 2:
                                    if(gradientG < -0.5 && gradientG >= -2 && stick_end.x < stick_last.x){
                                        poseState = 2;
                                    }else if(gradientG > 0 && stick_end.x < stick_last.x){
                                        poseState = 20;
                                    }else{
                                        poseState = -1;
                                    }
                                    break;
                            case 20:
                                    if(gradientG > 0 && stick_end.x < stick_last.x){
                                        poseState = 20;
                                    }else if(gradientG < -0.5 && stick_end.x < stick_last.x){
                                        poseState = 200;
                                    } else{
                                        poseState = -1;
                                    }
                                    break;
                            case 200:
                                    if(gradientG < -0.5 && stick_end.x < stick_last.x){
                                        poseState = 200;
                                    }else if(pow(stick_last.x - stick_end.x, 2) + pow(stick_last.y - stick_end.y, 2) <= MIN_VELOCITY){
                                        stallState = (stallState + 1) % DURATION;
                                        if(stallState == 0  && gradientK > 0 && RWristx < RElbowx){
                                            std::cout<<"On detection : W " << std::endl;
                                            poseState = -1;
                                        }
                                    }else{
                                        poseState = -1;
                                        stallState = 0;
                                    }
                                    break;

                            case 31:
                                    if((gradientG < -2 || gradientG > 2) && stick_end.y > stick_last.y){
                                        poseState = 31;
                                    }else if(gradientG >= -0.5 && gradientG < 0.5 && stick_end.x < stick_last.x){
                                        poseState = 310;
                                    }else{
                                        poseState = -1;
                                    }
                                    break;
                            case 310:
                                    if(gradientG >= -0.5 && gradientG < 0.5 && stick_end.x < stick_last.x){
                                        poseState = 310;
                                    }else if((gradientG < -2 || gradientG > 2) && stick_end.y < stick_last.y){
                                        poseState = 3100;
                                    }else{
                                        poseState = -1;
                                    }
                                    break;

                            case 3100:
                                    if((gradientG < -2 || gradientG > 2) && stick_end.y < stick_last.y){
                                        poseState = 3100;
                                    }else if(pow(stick_last.x - stick_end.x, 2) + pow(stick_last.y - stick_end.y, 2) <= MIN_VELOCITY){
                                        stallState = (stallState + 1) % DURATION;
                                        if(stallState == 0  && gradientK > 0 && RWristx < RElbowx){
                                            std::cout<<"On detection : 口 " << std::endl;
                                            poseState = -1;
                                        }
                                    }else{
                                        poseState = -1;
                                        stallState = 0;
                                    }
                                    break;

                            case 32:
                                    if(gradientG >= -0.5 && gradientG < 0.5 && stick_end.x < stick_last.x){
                                        poseState = 32;
                                    }else if((gradientG < -2 || gradientG > 2) && stick_end.y > stick_last.y){
                                        poseState = 320;
                                    }else{
                                        poseState = -1;
                                    }
                                    break;
                            case 320:
                                    if((gradientG < -2 || gradientG > 2) && stick_end.y > stick_last.y){
                                        poseState = 320;
                                    }else if(gradientG >= -0.5 && gradientG < 0.5 && stick_end.x > stick_last.x){
                                        poseState = 3200;
                                    }else{
                                        poseState = -1;
                                    }
                                    break;
                            case 3200:
                                    if(gradientG >= -0.5 && gradientG < 0.5 && stick_end.x > stick_last.x){
                                        poseState = 3200;
                                    }else if(pow(stick_last.x - stick_end.x, 2) + pow(stick_last.y - stick_end.y, 2) <= MIN_VELOCITY){
                                        stallState = (stallState + 1) % DURATION;
                                        if(stallState == 0  && gradientK > 0 && RWristx < RElbowx){
                                            std::cout<<"On detection : 口 " << std::endl;
                                            poseState = -1;
                                        }
                                    }else{
                                        poseState = -1;
                                        stallState = 0;
                                    }
                                    break;
                            case 4:
                                    if(pow(stick_last.x - stick_end.x, 2) + pow(stick_last.y - stick_end.y, 2) <= MIN_VELOCITY){
                                        stallState = (stallState + 1) % DURATION;
                                        if(stallState == 0  && gradientK > 0 && RWristx < RElbowx){
                                            std::cout<<"On detection : O " << std::endl;
                                            poseState = -1;
                                        }
                                    }else{
                                        poseState = 4;
                                    }
                                    break;

                            default:
                                    if(pow(stick_last.x - stick_end.x, 2) + pow(stick_last.y - stick_end.y, 2) <= MIN_VELOCITY){
                                        stallState = (stallState + 1) % DURATION;
                                        if(stallState == 0 && poseState == -1 && gradientK < 0 && RWristx > RElbowx){
                                            poseState = 1;
                                        }
                                    }else{
                                            poseState = -1;
                                    }
                        }
                        



                        // double y_offset,x_offset;
                        // // double stick_scale = 2;
                        // vector<cv::Vec4f> lines;
                        // cv::Mat crop_stick_o;
                        // cv::Mat crop_stick;
                        // cv::Rect area;
                        // // do {
                        //     // stick_scale += 1;
                        //     y_offset = STICK_RELATIVE_LENGTH * (RWristy - RElbowy);
                        //     x_offset = STICK_RELATIVE_LENGTH * (RWristx - RElbowx);
                        //     if(RWristy + y_offset < 0 ) y_offset = 1 - RWristy;
                        //     if(RWristy + y_offset > cvOutputData.rows ) y_offset = cvOutputData.rows - RWristy - 1;
                        //     if(RWristx + x_offset < 0 ) x_offset = 1 - RWristx;
                        //     if(RWristx + x_offset > cvOutputData.cols ) x_offset = cvOutputData.cols - RWristx - 1;


                        //     if (x_offset > 0){
                        //         if(y_offset >0){
                        //             area = cv::Rect(int(RWristx), int(RWristy),
                        //             abs(int(x_offset)),
                        //             abs(int(y_offset)));
                        //         }else{
                        //             area = cv::Rect(int(RWristx), int(RWristy + y_offset),
                        //             abs(int(x_offset)),
                        //             abs(int(y_offset)));
                        //         }
                        //     }else{
                        //         if(y_offset >0){
                        //             area = cv::Rect(int(RWristx + x_offset), int(RWristy),
                        //             abs(int(x_offset)),
                        //             abs(int(y_offset)));
                        //         }else{
                        //             area = cv::Rect(int(RWristx + x_offset), int(RWristy + y_offset),
                        //             abs(int(x_offset)),
                        //             abs(int(y_offset)));
                        //         }
                        //     }
                        //     cv::circle(cvOutputData, cv::Point(
                        //     (int)(RWristx + x_offset),
                        //     (int)(RWristy + y_offset)), 
                        //     5, cv::Scalar(0, 0, 255), -1);

                        //     crop_stick_o = cvOutputData(area);
                        //     if(crop_stick_o.empty()){
                        //         break;
                        //     }

                        //     crop_stick_o.copyTo(crop_stick);
                            

                        //     cv::cvtColor(crop_stick, crop_stick, cv::COLOR_RGB2GRAY);
                        //     cv::Canny(crop_stick, crop_stick, 80, 180, 3, false);
                        //     cv::threshold(crop_stick, crop_stick, 170, 255, cv::THRESH_BINARY);
                            
                            
                        //     // cv::HoughLines(crop_stick, lines, 1, CV_PI / 180, 50, 0, 0);
                        //     cv::HoughLinesP( crop_stick, lines, 1, CV_PI/180, 30, 30, 10 );

                        // // }while(lines.size()<1 && stick_scale < 4);
                        // vector<int>stick_pointy(2);
                        // cv::Point2f kalman_stick;
                        // if(lines.size()>0){
                        //     vector<int>stick_end(2);
                        //     if(x_offset >0 ){
                        //         // cv::circle(crop_stick_o, cv::Point( lines[0][2], lines[0][3]), 5, cv::Scalar(255, 0, 0), -1);

                        //         stick_end[0] = lines[0][2];
                        //         stick_end[1] = lines[0][3];
                        //     }else{
                        //         // cv::circle(crop_stick_o, cv::Point( lines[0][0], lines[0][1]), 5, cv::Scalar(255, 0, 0), -1);
                        //         stick_end[0] = lines[0][0];
                        //         stick_end[1] = lines[0][1];
                        //     }
                        //     stick_end[0] += area.x;
                        //     stick_end[1] += area.y;
                        //     cv::circle(cvOutputData, cv::Point(stick_end[0],stick_end[1]),
                        //                                  5, cv::Scalar(0, 255, 0), -1);
                        //     kalman_stick = kalman.run(float(stick_end[0]),
                        //                             float(stick_end[1]));
                        //     if(pow(kalman_stick.x - stick_end[0],2) + pow(kalman_stick.y - stick_end[1],2) <= MIN_VELOCITY){
                        //         if(stick_point.size() >= EXCLUTION){

                        //             stick_point.clear();
                        //         }else{
                        //             stick_point.clear();
                        //         }
                        //     }
                        // }else{
                        //     double s = 5./6.;
                        //     kalman_stick = kalman.run(s*(RWristx + x_offset),
                        //                             s*(RWristy + y_offset));
                        // }
                        // stick_pointy[0] = int(kalman_stick.x);
                        // stick_pointy[1] = int(kalman_stick.y);
                        // stick_point.push_back(stick_pointy);
                        

                        // cv::circle(cvOutputData, kalman_stick, 5, cv::Scalar(255, 0, 0), -1);
                        // cv::imwrite("./build/a"+std::to_string(fcount)+"hello.jpg", cvOutputData);
                        // //  if(!crop_stick_o.empty()){
                        // //     cv::imwrite("./build/b"+std::to_string(fcount)+"hello.jpg", crop_stick_o);
                        // // }
                         
                        //  fcount ++;

                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};

void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // producerType
        op::ProducerType producerType;
        op::String producerString;
        std::tie(producerType, producerString) = op::flagsToProducer(
            op::String(FLAGS_image_dir), op::String(FLAGS_video), op::String(FLAGS_ip_camera), FLAGS_camera,
            FLAGS_flir_camera, FLAGS_flir_camera_index);
        // cameraSize
        const auto cameraSize = op::flagsToPoint(op::String(FLAGS_camera_resolution), "-1x-1");
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1 || FLAGS_flir_camera);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Initializing the user custom classes
        // Processing
        auto wUserPostProcessing = std::make_shared<WUserPostProcessing>();
        // Add custom processing
        const auto workerProcessingOnNewThread = true;
        opWrapper.setWorker(op::WorkerType::PostProcessing, wUserPostProcessing, workerProcessingOnNewThread);

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, op::String(FLAGS_model_folder), heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
            op::String(FLAGS_prototxt_path), op::String(FLAGS_caffemodel_path),
            (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Producer (use default to disable any input)
        const op::WrapperStructInput wrapperStructInput{
            producerType, producerString, FLAGS_frame_first, FLAGS_frame_step, FLAGS_frame_last,
            FLAGS_process_real_time, FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat,
            cameraSize, op::String(FLAGS_camera_parameter_path), FLAGS_frame_undistort, FLAGS_3d_views};
        opWrapper.configure(wrapperStructInput);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port)};
        opWrapper.configure(wrapperStructOutput);
        // GUI (comment or use default argument to disable any visual output)
        const op::WrapperStructGui wrapperStructGui{
            op::flagsToDisplayMode(FLAGS_display, FLAGS_3d), !FLAGS_no_gui_verbose, FLAGS_fullscreen};
        opWrapper.configure(wrapperStructGui);
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int openPose()
{
    try
    {
        op::opLog("Starting OpenPose ...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Configure OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper;
        configureWrapper(opWrapper);

        // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.exec();

        // Measuring total time
        op::printTime(opTimer, "OpenPose successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return successful message
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPose
    return openPose();
}
