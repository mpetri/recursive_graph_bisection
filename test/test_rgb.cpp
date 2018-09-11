#include <gtest/gtest.h>

#include <rgb/rgb.hpp>

TEST(compute_single_gain, single_term)
{
    uint32_t terms[] = { 3 };
    docid_node doc{ 0, terms, 1 };
    auto doc_ref = std::ref(doc);
    std::vector<float> before = { 0.5, 1, 2, 3, 4, 5 };
    std::vector<float> after = { 1.5, 1, 1, 2, 3, 4 };
    auto move_gain = compute_single_gain(&doc_ref, before, after);

    ASSERT_FLOAT_EQ(1, move_gain.gain);
}

TEST(compute_single_gain, multi_term)
{
    uint32_t terms[] = { 1, 2, 3, 4, 5};
    docid_node doc{ 0, terms, 5 };
    auto doc_ref = std::ref(doc);
    std::vector<float> before = { 0.5, 1, 2, 3, 4, 5 };
    std::vector<float> after = { 1.5, 1, 1, 2, 3, 4 };
    auto move_gain = compute_single_gain(&doc_ref, before, after);

    ASSERT_FLOAT_EQ(4, move_gain.gain);
}

TEST(compute_single_gain, multi_term_negative)
{
    uint32_t terms[] = { 0, 1, 2, 3, 4, 5};
    docid_node doc{ 0, terms, 6 };
    auto doc_ref = std::ref(doc);
    std::vector<float> before = { 0.5, 1, 2, 3, 4, 5 };
    std::vector<float> after = { 1.5, 1, 1, 2, 3, 4 };
    auto move_gain = compute_single_gain(&doc_ref, before, after);

    ASSERT_FLOAT_EQ(3, move_gain.gain);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
