#include "Matcher.hpp"

const float VP_DIM = 0.06;

//float getScale(cv::Mat& image1, cv::Mat& image2, glm::vec2 uv1, glm::vec2 uv2);

int main()
{
    Matcher matcher;
    
    namedWindow("compare",1);
    
    Mat Gala1 = imread("Textures/BLV_st_Jacques1.jpg");
    Mat Gala2 = imread("Textures/BLV_st_Jacques2.jpg");
    
    Vec2f uv = Vec2f(0.5,0.5);
    
//    Mat Gala1Cut = Gala1;
//    Mat Gala2Cut = Gala2;
    Mat Gala1Cut = Gala1(cvRect(Gala1.cols *uv[0] - 200, Gala1.rows *uv[1] - 100, 400,400));
    Mat Gala2Cut = Gala2(cvRect(Gala2.cols *uv[0] - 200, Gala2.rows *uv[1] - 100, 400,400));
    
    matcher.addTrainImage(Gala2Cut);
    
    vector<KeyPoint> keypoints;
    vector<vector<DMatch>> matches;
    
    matcher.match(Gala1Cut, matches, keypoints);
    
    Mat img_matches;
    drawMatches( Gala2Cut, matcher.getTrainImgKeyPoints(0), Gala1Cut, keypoints,
                                matches[0], img_matches, Scalar::all(-1), Scalar::all(-1),
                                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    //matcher.DrawAllBoundingBoxAndTrajectories(Gala1Cut, matches, keypoints);
    matcher.DrawAllBoundingBoxes(Gala1Cut, matches, keypoints);
    cout << matcher.getScale(Gala1Cut, matches[0], keypoints) << endl;
    
    imshow("compare",Gala1Cut);
    //imshow("compare1",Gala2Cut);
    imshow("compare2",img_matches);
    
    waitKey(0);
    
    return 0;
}
