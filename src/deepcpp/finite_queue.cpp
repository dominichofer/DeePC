#include "finite_queue.h"

void FiniteQueue::push_back(VectorXd value)
{
    data.push_back(std::move(value));
    if (data.size() > static_cast<std::size_t>(max_size))
        data.pop_front();
}

std::vector<VectorXd> FiniteQueue::get() const
{
    return std::vector<VectorXd>(data.begin(), data.end());
}

int FiniteQueue::size() const
{
    return data.size();
}

void FiniteQueue::clear()
{
    data.clear();
}
