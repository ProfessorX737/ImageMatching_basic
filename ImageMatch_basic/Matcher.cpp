//
//  Matcher.cpp
//  ImageMatching
//
//  Created by Xavier Poon on 3/09/2017.
//  Copyright © 2017 CreativityInk. All rights reserved.


#include "Matcher.hpp"

#define drawCross( img, center, color, d )\
line(img, Point(center.x - d, center.y - d), Point(center.x + d, center.y + d), color, 2, CV_AA, 0);\
line(img, Point(center.x + d, center.y - d), Point(center.x - d, center.y + d), color, 2, CV_AA, 0 )\

Matcher::Matcher() {
    
    Detector = SIFT::create();
    DesMatcher = DescriptorMatcher::create(DEFAULT_DESCRIPTOR_MATCHER);
    Extractor = FastFeatureDetector::create();
    NumTrainImages = 0;
    TrajectoryLen = DEFAULT_TRAJECTORY_LENGTH;
    MinHessian = DEFAULT_MIN_HESSIAN;
    Ratio = DEFAULT_RATIO;
    LineThickness = DEFAULT_LINE_THICKNESS;
    TrajectoryDecrement = DEFAULT_TRAJECTORY_DECREMENT;
}

Matcher::Matcher(int minHessian, float ratio) {
    Detector = SIFT::create();
    DesMatcher = DescriptorMatcher::create(DEFAULT_DESCRIPTOR_MATCHER);
    Extractor = FastFeatureDetector::create();
    NumTrainImages = 0;
    TrajectoryLen = DEFAULT_TRAJECTORY_LENGTH;
    MinHessian = minHessian;
    Ratio = ratio;
    LineThickness = DEFAULT_LINE_THICKNESS;
    TrajectoryDecrement = DEFAULT_TRAJECTORY_DECREMENT;
}

void Matcher::addTrainImage(Mat& image) {
    
    vector<KeyPoint> keypoints;
    Mat descriptor;
    
    //Detector->detectAndCompute(image, Mat(), keypoints, descriptor );
    Extractor->detect(image, keypoints);
    Detector->compute(image, keypoints, descriptor);
    
    TrainKeyPoints.push_back(keypoints);
    TrainDescriptors.push_back(descriptor);
    
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( image.cols, 0 );
    obj_corners[2] = cvPoint( image.cols, image.rows );
    obj_corners[3] = cvPoint( 0, image.rows );
    
    ObjectsCorners.push_back(obj_corners);
    ObjectSizes.push_back(image.size());
    Trajectory.emplace_back();
    PredictedTraj.emplace_back();
    KFs.emplace_back();
    Measurements.emplace_back();
    States.emplace_back();
    PredictCounts.emplace_back();
    
    InitKalman((int)(KFs.size()-1));
    
    NumTrainImages++;
}

void Matcher::addTrainImages(vector<Mat>& images) {
    for(int i = 0; i < (int)images.size(); i++) {
        addTrainImage(images[i]);
    }
}

vector<KeyPoint> Matcher::getTrainImgKeyPoints(int index) {
    return TrainKeyPoints[index];
}

Mat Matcher::getTrainImgDescription(int index) {
    return TrainDescriptors[index];
}

int Matcher::getNumTrainImgs() {
    return NumTrainImages;
}

vector<Point2f> Matcher::getObjectCorners(int index) {
    return ObjectsCorners[index];
}

void Matcher::SetRatioTestRatio(float ratio) {
    Ratio = ratio;
}

void Matcher::SetMinHessian(int minHessian) {
    MinHessian = minHessian;
}

void Matcher::SetTrajectoryLen(int len) {
    TrajectoryLen = len;
}

void Matcher::IncreaseRatio(float diff) {
    Ratio += diff;
}

void Matcher::DecreaseRatio(float diff) {
    Ratio -= diff;
}

// detects the keypoints and computes the descriptors in one function
void Matcher::match(Mat& queryImg,vector<DMatch>& outMatches, vector<KeyPoint>& outKeypoints, int trainIndex) {
    
    Mat targetDes;
    
    //Detector->detectAndCompute( queryImg, Mat(), outKeypoints, targetDes );
    Extractor->detect(queryImg, outKeypoints);
    Detector->compute(queryImg, outKeypoints, targetDes);
    
    // return a vector of matches where each vector element is a vector
    // containing two best match candidates
    std::vector<std::vector<cv::DMatch>> matches1;
    DesMatcher->knnMatch(TrainDescriptors[trainIndex], targetDes, matches1, 2);
    // repeat but from image 2 to image 1 for accuracy
    std::vector<std::vector<cv::DMatch>> matches2;
    DesMatcher->knnMatch(targetDes, TrainDescriptors[trainIndex], matches2, 2);
    
    //clean both matches using ratio test
    int removed = RatioTest(matches1);
    removed = RatioTest(matches2);
    
    // Remove non symmetrical matches --> extract the matches that are in agreement to both sets
    vector<DMatch> symMatches;
    SymmetryTest(matches1, matches2, outMatches);
}

// return a vector of matches where each vector element is a vector
void Matcher::match(Mat& queryImg, vector<DMatch>& outMatches, Mat& sceneDescriptor, int trainIndex) {
    
    // containing two best match candidates
    std::vector<std::vector<cv::DMatch>> matches1;
    DesMatcher->knnMatch(TrainDescriptors[trainIndex], sceneDescriptor, matches1, 2);
    // repeat but from image 2 to image 1 for accuracy
    std::vector<std::vector<cv::DMatch>> matches2;
    DesMatcher->knnMatch(sceneDescriptor, TrainDescriptors[trainIndex], matches2, 2);
    
    //clean both matches using ratio test
    int removed = RatioTest(matches1);
    removed = RatioTest(matches2);
    
    //Remove non symmetrical matches
    vector<DMatch> symMatches;
    SymmetryTest(matches1, matches2, outMatches);
    //    for(int i = 0; i < matches1.size(); i++) {
    //        outMatches.push_back(matches1[i][0]);
    //    }
    
}

void Matcher::match(Mat& queryImg,vector<vector<DMatch>>& outMatches,vector<KeyPoint>& outKeypoints) {
    
    Mat targetDes;
    vector<DMatch> matches_tmp;
    
    //Detector->detectAndCompute(queryImg, Mat(), outKeypoints, targetDes);
    Extractor->detect(queryImg, outKeypoints);
    Detector->compute(queryImg, outKeypoints, targetDes);
    
    for(int i = 0; i < NumTrainImages; i++) {
        matches_tmp.clear();
        match(queryImg, matches_tmp, targetDes, i);
        outMatches.push_back(matches_tmp);
    }
    
}

int Matcher::RatioTest(vector<vector<DMatch>> &matches) {
    int removed = 0;
    for(vector<vector<DMatch>>::iterator matchIt = matches.begin();
        matchIt != matches.end(); ++matchIt) {
        // if 2 matches in the vector are identified
        if(matchIt->size() > 1) {
            // check the distance ratio
            if(((*matchIt)[0].distance / (*matchIt)[1].distance) > Ratio) {
                matchIt->clear(); // remove match
                removed++;
            }
        }
        else { // this particular vector does not have 2 matches in it
            matchIt->clear(); // remove match
            removed++;
        }
    }
    return removed;
}

void Matcher::SymmetryTest(const vector<vector<DMatch>>& matches1,
                           const vector<vector<DMatch>>& matches2,
                           vector<DMatch>& symMatches) {
    
    // for all matches from the first image to the second image
    for(vector<vector<DMatch>>::const_iterator matchIt1 = matches1.begin();
        matchIt1 != matches1.end(); ++matchIt1) {
        
        // ignore deleted matches
        if(matchIt1->size() < 2) {
            continue;
        }
        // for all matches from the second image to the first image
        for(vector<vector<DMatch>>::const_iterator matchIt2 = matches2.begin();
            matchIt2 != matches2.end(); ++matchIt2) {
            
            
            if(matchIt2->size() < 2) {
                continue;
            }
            
            
            if ((*matchIt1)[0].queryIdx ==
                (*matchIt2)[0].trainIdx  &&
                (*matchIt2)[0].queryIdx ==
                (*matchIt1)[0].trainIdx) {
                
                symMatches.push_back(cv::DMatch((*matchIt1)[0].queryIdx,
                                                (*matchIt1)[0].trainIdx,
                                                (*matchIt1)[0].distance));
                break;
            }
        }
    }
    
}

float Matcher::distance(Point2f p1, Point2f p2) {
    
    float distX = (p1.x - p2.x) * (p1.x - p2.x);
    float distY = (p1.y - p2.y) * (p1.y - p2.y);
    
    return sqrtf(distX + distY);
}

void Matcher::getGoodKeyPoints(vector<KeyPoint>& sceneKeyPoints, vector<DMatch>& matches, vector<Point2f>& outGoodObjPts, vector<Point2f>& outGoodScenePts, int trainIndex) {
    
    getGoodKeyPoints(TrainKeyPoints[trainIndex], sceneKeyPoints, matches, outGoodObjPts, outGoodScenePts);
    
}

void Matcher::getGoodKeyPoints(vector<KeyPoint>& objKeyPoints, vector<KeyPoint>& sceneKeyPts, vector<DMatch>& matches, vector<Point2f>& outGoodObjPts, vector<Point2f>& outGoodScenePts) {

    for(int i = 0; i < matches.size(); i++) {
        // get good matching keypoint from object images and scene images
        outGoodObjPts.push_back(objKeyPoints[matches[i].queryIdx].pt);
        outGoodScenePts.push_back(sceneKeyPts[matches[i].trainIdx].pt);
    }
    
}

int Matcher::getSceneQuadCorners(Mat& scene, vector<DMatch>& matches, vector<KeyPoint>& sceneKeyPts, std::vector<Point2f>& scene_corners, int trainIndex) {

    std::vector<Point2f> objMatch_pts;
    std::vector<Point2f> sceneMatch_pts;
    
    for( int i = 0; i < matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        objMatch_pts.push_back(TrainKeyPoints[trainIndex][matches[i].queryIdx].pt);
        sceneMatch_pts.push_back(sceneKeyPts[matches[i].trainIdx].pt);
    }
    Mat H;
    if(!objMatch_pts.empty() && !sceneMatch_pts.empty()) {
        H = findHomography(objMatch_pts, sceneMatch_pts, CV_RANSAC);
    }
    
    // only consider bounding box if it is meaningful
    if(!H.empty() && matches.size() >= BOUNDING_BOX_MIN_MATCHES) {
        
        perspectiveTransform(ObjectsCorners[trainIndex], scene_corners, H);
        return 1; // success
    }
    
    return 0; // failure
}

float Matcher::getScale(Mat& scene, vector<DMatch>& matches, vector<KeyPoint>& sceneKeyPts, int trainIndex) {
    
    std::vector<Point2f> scene_corners;
    
    if(getSceneQuadCorners(scene, matches, sceneKeyPts, scene_corners, trainIndex)) {
        // transform into a regular rectangle by averages of corners
        int x1 = (scene_corners[0].x + scene_corners[3].x) / 2;
        int x2 = (scene_corners[1].x + scene_corners[2].x) / 2;
        int y1 = (scene_corners[0].y + scene_corners[1].y) / 2;
        int y2 = (scene_corners[3].y + scene_corners[2].y) / 2;
        
        int width = x2 - x1;
        int height = y2 - y1;
        
        float scaleX = (float)width / ObjectSizes[trainIndex].width;
        float scaleY = (float)height / ObjectSizes[trainIndex].height;
        
        float minScale = std::fmin(scaleX, scaleY);
        
        return minScale;
        
    } else {
        
        return 0.0;
    }
}

void Matcher::DrawBoundingBox(Mat& scene, vector<DMatch>& matches, vector<KeyPoint>& sceneKeyPts, int trainIndex, Scalar lineColor) {
    
    std::vector<Point2f> objMatch_pts;
    std::vector<Point2f> sceneMatch_pts;
    
    for( int i = 0; i < matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        objMatch_pts.push_back(TrainKeyPoints[trainIndex][matches[i].queryIdx].pt);
        sceneMatch_pts.push_back(sceneKeyPts[matches[i].trainIdx].pt);
    }
    Mat H;
    if(!objMatch_pts.empty() && !sceneMatch_pts.empty()) {
        H = findHomography(objMatch_pts, sceneMatch_pts, CV_RANSAC);
    }
    
    // only draw bounding box if it is meaningful
    if(!H.empty() && matches.size() >= BOUNDING_BOX_MIN_MATCHES) {
        
        std::vector<Point2f> scene_corners(4);
        perspectiveTransform(ObjectsCorners[trainIndex], scene_corners, H);
        
        line(scene, scene_corners[0], scene_corners[1], BOUNDING_BOX_COLOR, LineThickness);
        line(scene, scene_corners[1], scene_corners[2], BOUNDING_BOX_COLOR, LineThickness);
        line(scene, scene_corners[2], scene_corners[3], BOUNDING_BOX_COLOR, LineThickness);
        line(scene, scene_corners[3], scene_corners[0], BOUNDING_BOX_COLOR, LineThickness);
        
    }
    
}

void Matcher::DrawAllBoundingBoxes(Mat& scene, vector<vector<DMatch>>& matches, vector<KeyPoint>& sceneKeyPts, Scalar lineColor) {
    
    for(int i = 0; i < NumTrainImages; i++) {
        DrawBoundingBox(scene, matches[i], sceneKeyPts, i);
    }
}

Point2f Matcher::getCentrePoint(vector<Point2f>& corners) {
    int width = corners[1].x - corners[0].x;
    int height = corners[3].y - corners[0].y;
    return Point2f(corners[0].x + width/2, corners[0].y + height/2);
}

vector<Point2f> Matcher::getCornersFromCentre(Point2f centre, Size size) {
    vector<Point2f> corners = vector<Point2f>(4);
    corners[0] = Point2f(centre.x - size.width/2,centre.y + size.height/2);
    corners[1] = Point2f(centre.x + size.width/2,centre.y + size.height/2);
    corners[2] = Point2f(centre.x + size.width/2,centre.y - size.height/2);
    corners[3] = Point2f(centre.x - size.width/2,centre.y - size.height/2);
    return corners;
}

void Matcher::DrawBoundingBoxAndTrajectories(Mat& scene, vector<DMatch>& matches, vector<KeyPoint>& sceneKeyPts, int trainIndex, Scalar lineColor) {
    
    if( scene.type() == CV_8U ) {
        cvtColor( scene, scene, COLOR_GRAY2BGR );
    }
    
    Scalar predBoundingBoxColor = PREDICTED_BOUNDING_BOX_COLOR;
    
    std::vector<Point2f> objMatch_pts;
    std::vector<Point2f> sceneMatch_pts;
    
    if(PredictCounts[trainIndex] < PREDICTION_TOLERANCE + ERRONEOUS_PREDICTION_TOLERANCE) {
        Point2f predictPt = KalmanPredict(trainIndex);
        if(PredictedTraj[trainIndex].size() < TrajectoryLen) {
            PredictedTraj[trainIndex].push_back(predictPt);
        } else {
            PredictedTraj[trainIndex].erase(PredictedTraj[trainIndex].begin());
            PredictedTraj[trainIndex].push_back(predictPt);
        }
        if(PredictCounts[trainIndex] > PREDICTION_TOLERANCE) {
            predBoundingBoxColor = RED;
        }
        PredictCounts[trainIndex]++;
        
    } else {
        PredictedTraj[trainIndex].clear();
    }
    
    for( int i = 0; i < matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        objMatch_pts.push_back(TrainKeyPoints[trainIndex][matches[i].queryIdx].pt);
        sceneMatch_pts.push_back(sceneKeyPts[matches[i].trainIdx].pt);
    }
    Mat H;
    if(!objMatch_pts.empty() && !sceneMatch_pts.empty()) {
        H = findHomography(objMatch_pts, sceneMatch_pts, CV_RANSAC);
    }
    
    // only draw bounding box if it is meaningful
    if(!H.empty() && matches.size() >= BOUNDING_BOX_MIN_MATCHES) {
        
        std::vector<Point2f> scene_corners(4);
        perspectiveTransform(ObjectsCorners[trainIndex], scene_corners, H);
        
        line(scene, scene_corners[0], scene_corners[1], BOUNDING_BOX_COLOR, LineThickness);
        line(scene, scene_corners[1], scene_corners[2], BOUNDING_BOX_COLOR, LineThickness);
        line(scene, scene_corners[2], scene_corners[3], BOUNDING_BOX_COLOR, LineThickness);
        line(scene, scene_corners[3], scene_corners[0], BOUNDING_BOX_COLOR, LineThickness);
        
        // update the trajectory
        Point2f measPt = getCentrePoint(scene_corners);
        
        if((int)Trajectory[trainIndex].size() < TrajectoryLen) {
            Trajectory[trainIndex].push_back(measPt);
        } else {
            Trajectory[trainIndex].erase(Trajectory[trainIndex].begin());
            Trajectory[trainIndex].push_back(measPt);
        }
        
        Point2f statePt = KalmanCorrect(trainIndex, measPt);
        PredictCounts[trainIndex] = 0;
        
    } else {
        Trajectory[trainIndex].clear();
        if(!PredictedTraj[trainIndex].empty()) {
            vector<Point2f> corners = getCornersFromCentre(PredictedTraj[trainIndex].back(), ObjectSizes[trainIndex]);
            line(scene, corners[0], corners[1], predBoundingBoxColor, LineThickness);
            line(scene, corners[1], corners[2], predBoundingBoxColor, LineThickness);
            line(scene, corners[2], corners[3], predBoundingBoxColor, LineThickness);
            line(scene, corners[3], corners[0], predBoundingBoxColor, LineThickness);
        }
    }
    
    // only draw trajectory if it is meaningful
    if(matches.size() >= TRAJECTORY_MIN_MATCHES) {
        for(int i = 0; i < (int)Trajectory[trainIndex].size() - 1; i++) {
            float thickness = (float)(sqrt(float(i + 1)/TrajectoryDecrement) * LineThickness);
            line(scene, Trajectory[trainIndex][i], Trajectory[trainIndex][i+1], TRAJECTORY_COLOR, thickness);
        }
    } else {
        Trajectory[trainIndex].clear();
    }
    
    // draw predicted trajectory
    for(int i = 0; i < (int)PredictedTraj[trainIndex].size() - 1; i++) {
        float thickness = (float)(sqrt(float(i + 1)/TrajectoryDecrement) * LineThickness);
        line(scene, PredictedTraj[trainIndex][i], PredictedTraj[trainIndex][i+1], PREDICTED_TRAJECTORY_COLOR, thickness);
    }
}

void Matcher::DrawAllBoundingBoxAndTrajectories(Mat& scene, vector<vector<DMatch>>& matches, vector<KeyPoint>& sceneKeyPts, Scalar lineColor) {
    
    for(int i = 0; i < NumTrainImages; i++) {
        DrawBoundingBoxAndTrajectories(scene, matches[i], sceneKeyPts,i);
    }
    
}

void Matcher::InitKalman(int objectIndex, Point2f startPoint)
{
    // Instantate Kalman Filter with
    // 4 dynamic parameters and 2 measurement parameters,
    // where my measurement is: 2D location of object,
    // and dynamic is: 2D location and 2D velocity.
    KFs[objectIndex].init(4, 2, 0);
    
    KFs[objectIndex].transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0, 0,1,0,1, 0,0,1,0, 0,0,0,1);
    Measurements[objectIndex] = Mat_<float>::zeros(2,1);
    
    setIdentity(KFs[objectIndex].measurementMatrix);
    setIdentity(KFs[objectIndex].processNoiseCov, Scalar::all(1e-2));
    setIdentity(KFs[objectIndex].measurementNoiseCov, Scalar::all(10));
    setIdentity(KFs[objectIndex].errorCovPost, Scalar::all(.1));
}

Point2f Matcher::KalmanPredict(int objectIndex)
{
    Mat prediction = KFs[objectIndex].predict();
    Point2f predictPt(prediction.at<float>(0),prediction.at<float>(1));
    
    KFs[objectIndex].statePre.copyTo(KFs[objectIndex].statePost);
    KFs[objectIndex].errorCovPre.copyTo(KFs[objectIndex].errorCovPost);
    
    return predictPt;
}

Point2f Matcher::KalmanCorrect(int objectIndex, Point2f correctPoint)
{
    Measurements[objectIndex](0) = correctPoint.x;
    Measurements[objectIndex](1) = correctPoint.y;
    
    Mat estimated = KFs[objectIndex].correct(Measurements[objectIndex]);
    Point2f statePt(estimated.at<float>(0),estimated.at<float>(1));
    
    return statePt;
}

void Matcher::ComputeAllKalmanInitPositions(Mat& scene) {
    
    for(int i = 0; i < NumTrainImages; i++) {
        
        vector<KeyPoint> sceneKps;
        vector<DMatch> matches;
        
        std::vector<Point2f> objMatch_pts;
        std::vector<Point2f> sceneMatch_pts;
        
        match(scene, matches, sceneKps, i);
        
        for( int j = 0; j < matches.size(); j++ )
        {
            objMatch_pts.push_back(TrainKeyPoints[i][matches[j].queryIdx].pt);
            sceneMatch_pts.push_back(sceneKps[matches[j].trainIdx].pt);
        }
        
        Mat H;
        if(!objMatch_pts.empty() && !sceneMatch_pts.empty()) {
            H = findHomography(objMatch_pts, sceneMatch_pts, CV_RANSAC);
        }
        
        if(!H.empty() && matches.size() > BOUNDING_BOX_MIN_MATCHES) {
            
            std::vector<Point2f> scene_corners(4);
            perspectiveTransform(ObjectsCorners[i], scene_corners, H);
            
            InitalObjectPositions.push_back(getCentrePoint(scene_corners));
            
        } else {
            InitalObjectPositions.emplace_back(scene.rows/2,scene.cols/2);
        }
        
        InitKalmanStartPos(i, InitalObjectPositions[i]);
        
    }
}

void Matcher::InitKalmanStartPos(int objectIndex, Point2f startPoint) {
    
    Measurements[objectIndex].at<float>(0, 0) = startPoint.x;
    Measurements[objectIndex].at<float>(0, 0) = startPoint.y;
    
    KFs[objectIndex].statePre.setTo(0);
    KFs[objectIndex].statePre.at<float>(0, 0) = startPoint.x;
    KFs[objectIndex].statePre.at<float>(1, 0) = startPoint.y;
    
    KFs[objectIndex].statePost.setTo(0);
    KFs[objectIndex].statePost.at<float>(0, 0) = startPoint.x;
    KFs[objectIndex].statePost.at<float>(1, 0) = startPoint.y;
}
