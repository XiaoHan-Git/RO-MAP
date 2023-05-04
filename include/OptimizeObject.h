/*
* Modification: SQ-SLAM
* Version: 1.0
* Created: 05/15/2022
* Author: Xiao Han
*/

#ifndef G2OOBJECT_H
#define G2OOBJECT_H

#include <iostream>
#include "dependencies/g2o/g2o/types/types_seven_dof_expmap.h"
#include "ObjectMap.h"

using namespace std;
using namespace ORB_SLAM2;

typedef Eigen::Matrix<double, 1, 1> Vector1d;
typedef Eigen::Matrix<double, 5, 1> Vector5d;

namespace g2o
{

    class VertexYaw : public BaseVertex<1,Vector1d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexYaw(){}

        virtual bool read(std::istream& is) {}
        virtual bool write(std::ostream& os) const {}

        virtual void setToOriginImpl() { _estimate = Vector1d(0);}

        virtual void oplusImpl(const double* update_)
        {
            Eigen::Map<const Vector1d> update(update_);
            //cout<<"update: "<<update<<endl;
            setEstimate(_estimate + update);
        }
    };

    class EdgeRotationLine : public BaseUnaryEdge<1,double,VertexYaw>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeRotationLine(){}

        virtual bool read(std::istream& is) {}
        virtual bool write(std::ostream& os) const {}

        void computeError()  {
            const VertexYaw* v1 = static_cast<const VertexYaw*>(_vertices[0]);
            double yaw = v1->estimate()(0);
            _error(0,0) = CalculateError(yaw);
        }

        double CalculateError(double yaw);

        Eigen::Vector3d axisPointPos;
        Eigen::Vector3d twobj;
        double lineAngle;
        Eigen::Vector2d centerUV;
        SE3Quat Tcw;
        double fx,fy,cx,cy;

    };

}

namespace ORB_SLAM2
{
class Frame;
class ObjectOptimizer
{
public:
    float static OptimizeRotation(const Object_Map& obj,const vector<vector<int>>& linesIdx,float IniYaw,cv::Mat twobj,const Frame& CurrentFrame);
};

}



#endif