#pragma once 
#define DOT_PROD_4X()          \
{                                                   \
    float32x4_t vecA, vecB0,vecB1,vecB2,vecB3;      \
    float32x4_t vecSum0,vecSum1,vecSum2,vecSum3;    \
    uint32_t blkCnt;                                \
    const float32_t *pA = q;                        \
    const float32_t *pB0 = k0;                      \
    const float32_t *pB1 = k1;                      \
    const float32_t *pB2 = k2;                      \
    const float32_t *pB3 = k3;                      \
    vecSum0 = vdupq_n_f32(0.0f);                    \
    vecSum1 = vdupq_n_f32(0.0f);                    \
    vecSum2 = vdupq_n_f32(0.0f);                    \
    vecSum3 = vdupq_n_f32(0.0f);                    \
                                                    \
    blkCnt = HEAD_SIZE >> 2U;                       \
    while (blkCnt > 0U)                             \
    {                                               \
                                                    \
        vecA = vld1q(pA);                           \
        pA += 4;                                    \
                                                    \
        vecB0 = vld1q(pB0);                         \
        pB0 += 4;                                   \
        vecSum0 = vfmaq(vecSum0, vecA, vecB0);      \
                                                    \
        vecB1 = vld1q(pB1);                         \
        pB1 += 4;                                   \
        vecSum1 = vfmaq(vecSum1, vecA, vecB1);      \
                                                    \
        vecB2 = vld1q(pB2);                         \
        pB2 += 4;                                   \
        vecSum2 = vfmaq(vecSum2, vecA, vecB2);      \
                                                    \
        vecB3 = vld1q(pB3);                         \
        pB3 += 4;                                   \
        vecSum3 = vfmaq(vecSum3, vecA, vecB3);      \
                                                    \
        blkCnt --;                                  \
    }                                               \
                                                    \
                                                    \
    blkCnt = HEAD_SIZE & 3;                         \
    if (blkCnt > 0U)                                \
    {                                               \
                                                    \
        mve_pred16_t p0 = vctp32q(blkCnt);          \
        vecA = vld1q(pA);                           \
                                                    \
        vecB0 = vld1q(pB0);                         \
        vecSum0 = vfmaq_m(vecSum0, vecA, vecB0, p0);\
                                                    \
        vecB1 = vld1q(pB1);                         \
        vecSum1 = vfmaq_m(vecSum1, vecA, vecB1, p0);\
                                                    \
        vecB2 = vld1q(pB2);                         \
        vecSum2 = vfmaq_m(vecSum2, vecA, vecB2, p0);\
                                                    \
        vecB3 = vld1q(pB3);                         \
        vecSum3 = vfmaq_m(vecSum3, vecA, vecB3, p0);\
    }                                               \
                                                    \
    score[0] = vecAddAcrossF32Mve(vecSum0);      \
    score[1] = vecAddAcrossF32Mve(vecSum1);      \
    score[2] = vecAddAcrossF32Mve(vecSum2);      \
    score[3] = vecAddAcrossF32Mve(vecSum3);      \
}