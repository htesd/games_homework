#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    model<< cos(rotation_angle), -sin(rotation_angle),0,0
        , sin(rotation_angle), cos(rotation_angle),0,0
        ,0,0,1,0
        ,0,0,0,1;

    std::cout <<"这是模型旋转矩阵，负责用于围绕z轴旋转"<<std::endl;

    std::cout<<model<<std::endl;

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.

    std::cout<<eye_fov<<std::endl;
    std::cout<<aspect_ratio<<std::endl;
    std::cout<<zNear<<std::endl;
    std::cout<<zFar<<std::endl;

    float scale=-zNear/zFar;
    float A=(zNear+zFar)/zFar;
    float B=-zNear;

    float l=-zNear* tan(eye_fov/2);
    float r=zNear* tan(eye_fov/2);
    float t=aspect_ratio*r;
    float b=aspect_ratio*l;

    Eigen::Matrix4f  trans;
    Eigen::Matrix4f  change;
    Eigen::Matrix4f  persp2ortho ;


    trans<<1,0,0,-(l+r)/2,
           0,1,0,-(t+b)/2,
           0,0,1,-(zNear+zFar),
           0,0,0,1;

    change<<2/(r-l),0,0,0,
            0,2/(t-b),0,0,
            0,0,2/(zNear-zFar),0,
            0,0,0,1;



    persp2ortho<<scale,0,0,0,
                0,scale,0,0,
                0,0,A,B,
                0,0,1/zFar,0;

    projection=change*trans*persp2ortho;

    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        std::cout<<"hahahaha"<<std::endl;
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};




    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    std::cout<<pos_id.pos_id<<std::endl;
    auto ind_id = r.load_indices(ind);
    std::cout<<ind_id.ind_id<<std::endl;
    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
