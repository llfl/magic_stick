
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
#define STICK_RELATIVE_LENGTH 1
#define DURATION 3
#define EXCLUTION 20
#define MIN_VELOCITY 20


int fin_state = 0;
int pattern_no = 0;

vector<vector<int> > stick_point;

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
            // User's post-processing (after OpenPose processing & before OpenPose outputs) here
                // datumPtr->cvOutputData: rendered frame with pose or heatmaps
                // datumPtr->poseKeypoints: Array<float> with the estimated pose
            if (datumsPtr != nullptr && !datumsPtr->empty())
            {

                for (auto& datumPtr : *datumsPtr)
                {
                    cv::Mat cvOutputData = OP_OP2CVMAT(datumPtr->cvOutputData);

                    const auto& poseKeypoints = datumPtr->poseKeypoints;
                    double RElbowx = poseKeypoints[{0, 3, 0}];
                    double RElbowy = poseKeypoints[{0, 3, 1}];
                    double RWristx = poseKeypoints[{0, 4, 0}];
                    double RWristy = poseKeypoints[{0, 4, 1}];
                    vector<int>stick_end(2);
                    stick_end[0] = (int)(RWristx + STICK_RELATIVE_LENGTH * (RWristx - RElbowx));
                    stick_end[1] = (int)(RWristy + STICK_RELATIVE_LENGTH * (RWristy - RElbowy));
                    if(stick_point.size() == 0)
                    {
                        stick_point.push_back(stick_end);
                    }
                    vector<int>stick_last = stick_point.back();
                    if(pow(stick_last[0] - stick_end[0],2) + pow(stick_last[1] - stick_end[1],2) <= MIN_VELOCITY)
                    {
                        fin_state = (fin_state + 1) % DURATION;
                        if (fin_state == 0 && stick_point.size() >= EXCLUTION)
                        {
                            cv::Mat stick_pattern(cvOutputData.rows, cvOutputData.cols, CV_8UC1, 255);
                            int minx, maxx, miny, maxy;
                            minx = stick_point[0][0];
                            maxx = stick_point[0][0];
                            miny = stick_point[0][1];
                            maxy = stick_point[0][1];

                            for (int i = 1; i<stick_point.size(); ++i )
                            {
                                // stick_pattern.at<cv::Vec3b>(stick_point[i][0],stick_point[i][1]) = 0;
                                if(minx>stick_point[i][0])minx = stick_point[i][0];
                                if(maxx<stick_point[i][0])maxx = stick_point[i][0];
                                if(miny>stick_point[i][1])miny = stick_point[i][1];
                                if(maxy<stick_point[i][1])maxy = stick_point[i][1];

                                cv::Point a(stick_point[i-1][0],stick_point[i-1][1]);
                                cv::Point b(stick_point[i][0],stick_point[i][1]);
                                cv::line(stick_pattern, a, b, 0, 10);
                            }
                            int long_side = maxx-minx;
                            if(long_side < maxy-miny) long_side = maxy - miny;
                            cv::Rect area(minx-16, miny-16 , long_side+16,long_side+16);
                            // cv::Mat resize_pattern = stick_pattern(area);
                            cv::Mat crop_pattern = stick_pattern(area);
                            cv::Mat resize_pattern(28, 28, CV_8UC1);
                            cv::resize(crop_pattern, resize_pattern, cv::Size(28,28));
                            cv::Mat flip_pattern;
                            cv::flip(resize_pattern,flip_pattern, 1);
                            cv::Mat float_pattern;
                            flip_pattern.convertTo(float_pattern, CV_32FC1);
                            // cv::Mat normalized_pattern;
                            // cv::subtract(float_pattern, mean_, normalized_pattern);
                            pattern_no ++;
                            // cv::imshow(std::to_string(pattern_no), resize_pattern);
                            cv::imwrite(std::to_string(pattern_no)+"hello2.jpg", flip_pattern);

                            // caffe::Caffe::set_mode(caffe::Caffe::GPU);
                            // caffe::Net<float> lenet("models/lenet.prototxt",caffe::TEST);
                            // lenet.CopyTrainedLayersFrom("models/lenet_iter_10000.caffemodel");
                            // caffe::Blob<float> *input_ptr = lenet.input_blobs()[0];
                            // input_ptr->Reshape(1,1,28,28);

                            // caffe::Blob<float> *output_ptr= lenet.output_blobs()[0];
                            // output_ptr->Reshape(1,10,1,1);

                            // input_ptr->set_cpu_data(reinterpret_cast<float*>(float_pattern.data));
                            // lenet.Forward();
                            // const float* begin = output_ptr->gpu_data();
    

                            // int index=0;
                            // for(int i=1;i<10;i++)
                            // {
                            //     if(begin[index]<begin[i])
                            //         index=i;
                            // }

                            cv::String modelTxt = "models/lenet.prototxt";
                            cv::String modelBin = "models/lenet_iter_10000.caffemodel";
                            cv::dnn::Net net;
                            net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
                            cv::Mat inputBlob = cv::dnn::blobFromImage(crop_pattern,1.0f,cv::Size(28,28),128);
                            net.setInput(inputBlob, "data");
                            std::vector<float> prob = std::vector<float>(net.forward("prob"));
                            auto predno = max_element(prob.begin(), prob.end());
                            std::cout<<"I guess the pattern is " << predno - prob.begin()<< std::endl;
                            op::opLog("over! ", op::Priority::High);
                            // std::cout<<""<<std::endl;
                            stick_point.clear();
                        }
                    }
                    else if(fin_state == 0 && stick_point.size() < EXCLUTION)
                    {
                        stick_point.clear();
                    }
                    else{
                        stick_point.push_back(stick_end);
                    }
                    // stick_point.push_back(stick_end);

                    
                    cv::Point current_stick_end(stick_end[0],stick_end[1]);
                    cv::circle(cvOutputData, current_stick_end, 5, cv::Scalar(0, 0, 255), -1);
                    for (int i = 1; i<stick_point.size(); ++i )
                    {
                        cv::Point a(stick_point[i-1][0],stick_point[i-1][1]);
                        cv::Point b(stick_point[i][0],stick_point[i][1]);
                        cv::line(cvOutputData, a, b, cv::Scalar(0, 255, 0), 10);
                    }
                    // cv::bitwise_not(cvOutputData, cvOutputData);
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

int openPoseDemo()
{
    try
    {
        op::opLog("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Configure OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper;
        configureWrapper(opWrapper);

        // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.exec();

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

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

    // Running openPoseDemo
    return openPoseDemo();
}
