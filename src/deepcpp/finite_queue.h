#include <deque>
#include <Eigen/Dense>

using Eigen::VectorXd;

class FiniteQueue
{
    std::deque<VectorXd> data;
    int max_size;
public:
    FiniteQueue(int max_size) : max_size(max_size) {}

    void push_back(VectorXd);
    std::vector<VectorXd> get() const;
    int size() const;
    void clear();
};
