//
//  Matcher.hpp
//  ImageMatching
//
//  Created by Xavier Poon on 3/09/2017.
//  Copyright © 2017 CreativityInk. All rights reserved.
//

#ifndef Matcher_hpp
#define Matcher_hpp

#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define DEFAULT_RATIO 0.7f
#define DEFAULT_MIN_HESSIAN 400
#define DEFAULT_TRAJECTORY_LENGTH 30
#define DEFAULT_LINE_THICKNESS 2
#define DEFAULT_TRAJECTORY_DECREMENT 6
#define TRAJECTORY_MIN_MATCHES 10
#define BOUNDING_BOX_MIN_MATCHES 6
#define PREDICTION_TOLERANCE 10
#define ERRONEOUS_PREDICTION_TOLERANCE 10

#define DEFAULT_DESCRIPTOR_MATCHER "FlannBased"
#define DEFAULT_DESCRIPTOR_EXTRACTOR "SURF"

// some common BGR colors defined
#define GREEN Scalar(0,255,0)
#define YELLOW Scalar(0,255,255)
#define RED Scalar(0,0,255)
#define BLUE Scalar(255,0,0)

#define BOUNDING_BOX_COLOR GREEN
#define PREDICTED_BOUNDING_BOX_COLOR YELLOW

#define TRAJECTORY_COLOR GREEN
#define PREDICTED_TRAJECTORY_COLOR YELLOW

class Matcher {
private:
    
    Ptr<FeatureDetector> Detector;
    Ptr<FastFeatureDetector> Extractor;
    Ptr<DescriptorMatcher> DesMatcher;
    vector<vector<KeyPoint>> TrainKeyPoints;
    vector<Mat> TrainDescriptors;
    vector<Mat> TrainImages;
    vector<vector<Point2f>> ObjectsCorners;
    vector<Size> ObjectSizes;
    vector<vector<Point2f>> Trajectory;
    vector<vector<Point2f>> PredictedTraj;
    vector<Point2f> InitalObjectPositions;
    
    vector<KalmanFilter> KFs;
    vector<cv::Mat_<float>> Measurements;
    vector<Mat_<float>> States; // (x, y, Vx, Vy)
    
    int NumTrainImages;
    float Ratio;
    int MinHessian;
    int TrajectoryLen;
    int LineThickness;
    int TrajectoryDecrement;
    vector<int> PredictCounts;
    //private functions
    void match(Mat& queryImg,vector<DMatch>& outMatches, Mat& sceneDescriptor, int trainIndex = 0);
    
public:
    
    Matcher();
    Matcher(int minHessian, float ratio);
    
    void addTrainImage(Mat& image);
    void addTrainImages(vector<Mat>& images);
    
    vector<KeyPoint> getTrainImgKeyPoints(int index);
    Mat getTrainImgDescription(int index);
    int getNumTrainImgs();
    vector<Point2f> getObjectCorners(int index);
    
    void SetRatioTestRatio(float ratio);
    void SetTrajectoryLen(int len);
    void SetMinHessian(int minHessian);
    void IncreaseRatio(float diff);
    void DecreaseRatio(float diff);
    
    // match object with scene image
    void match(Mat& queryImg, vector<DMatch>& outMatches,vector<KeyPoint>& outKeypoints,int trainIndex = 0);
    // match multiple objects with scene image
    void match(Mat& queryImg,vector<vector<DMatch>>& outMatches,vector<KeyPoint>& outKeypoints);
    
    int RatioTest(vector<vector<DMatch>> &matches);
    void SymmetryTest(const vector<vector<DMatch>>& matches1,
                      const vector<vector<DMatch>>& matches2,
                      vector<DMatch>& symMatches);
    
    void DrawBoundingBox(Mat& scene, vector<DMatch>& matches, vector<KeyPoint>& sceneKeyPts, int trainIndex = 0, Scalar lineColor = BOUNDING_BOX_COLOR);
    void DrawAllBoundingBoxes(Mat& scene, vector<vector<DMatch>>& matches, vector<KeyPoint>& sceneKeyPts, Scalar lineColor = BOUNDING_BOX_COLOR);
    void DrawBoundingBoxAndTrajectories(Mat& scene, vector<DMatch>& matches, vector<KeyPoint>& sceneKeyPts, int trainIndex = 0, Scalar lineColor = BOUNDING_BOX_COLOR);
    void DrawAllBoundingBoxAndTrajectories(Mat& scene, vector<vector<DMatch>>& matches, vector<KeyPoint>& sceneKeyPts, Scalar lineColor = BOUNDING_BOX_COLOR);
    
    Point2f getCentrePoint(vector<Point2f>& corners);
    vector<Point2f> getCornersFromCentre(Point2f centre,Size size);
    
    // Kalman functions
    void InitKalman(int objectIndex, Point2f startPoint = Point2f(100.0f,100.0f));
    //void SetKalmanStartPos(Point2f startPoint);
    Point2f KalmanPredict(int trainIndex);
    Point2f KalmanCorrect(int trainIndex, Point2f correctPoint);
    
    void ComputeAllKalmanInitPositions(Mat& scene);
    void InitKalmanStartPos(int trainIndex, Point2f startPoint);
    
    float distance(Point2f p1, Point2f p2);
    
    void getGoodKeyPoints(vector<KeyPoint>& sceneKeyPoints, vector<DMatch>& matches, vector<Point2f>& outGoodObjPts, vector<Point2f>& outGoodScenePts, int trainIndex = 0);
    
    // new
    void getGoodKeyPoints(vector<KeyPoint>& objKeyPoints, vector<KeyPoint>& sceneKeyPts, vector<DMatch>& matches, vector<Point2f>& outGoodObjPts, vector<Point2f>& outGoodScenePts);
    
    // new
    int getSceneQuadCorners(Mat& scene, vector<DMatch>& matches, vector<KeyPoint>& sceneKeyPts, std::vector<Point2f>& scene_corners, int trainIndex);
    
    // new, calls getSceneQuadCorners
    float getScale(Mat& scene, vector<DMatch>& matches, vector<KeyPoint>& sceneKeyPts, int trainIndex = 0);
    
};


#endif /* Matcher_hpp */
