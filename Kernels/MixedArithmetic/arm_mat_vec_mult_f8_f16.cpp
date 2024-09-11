#include "kernels.h"

#include <math.h>
#include "arm_math_types_f16.h"
#include "arm_helium_utils.h"


void arm_mat_vec_mult_f8_f16(
  const arm_matrix_instance_f8  *pSrcMat,
  const float16_t                 *pSrcVec,
  float16_t                       *pDstVec)
{
    uint32_t         numRows = pSrcMat->numRows;
    uint32_t         numCols = pSrcMat->numCols;
    const float8_t *pSrcA = pSrcMat->pData;
    const float8_t *pInA0;
    const float8_t *pInA1;
    float16_t       *px;
    int32_t          row;
    uint32_t         blkCnt;           /* loop counters */

    row = numRows;
    px = pDstVec;

    /*
     * compute 4 rows in parallel
     */
    while (row >= 4)
    {
        const float8_t     *pInA2, *pInA3;
        float8_t const    *pSrcA0Vec, *pSrcA1Vec, *pSrcA2Vec, *pSrcA3Vec;
        float16_t const *pInVec;
        f16x8_t            vecIn1, vecIn2,acc0, acc1, acc2, acc3;
        float16_t const     *pSrcVecPtr = pSrcVec;
        int8x16_t tmp;

        /*
         * Initialize the pointers to 4 consecutive MatrixA rows
         */
        pInA0 = pSrcA;
        pInA1 = pInA0 + numCols;
        pInA2 = pInA1 + numCols;
        pInA3 = pInA2 + numCols;
        /*
         * Initialize the vector pointer
         */
        pInVec =  pSrcVecPtr;
        /*
         * reset accumulators
         */
        acc0 = vdupq_n_f16(0.0f);
        acc1 = vdupq_n_f16(0.0f);
        acc2 = vdupq_n_f16(0.0f);
        acc3 = vdupq_n_f16(0.0f);

        pSrcA0Vec = pInA0;
        pSrcA1Vec = pInA1;
        pSrcA2Vec = pInA2;
        pSrcA3Vec = pInA3;

        blkCnt = numCols >> 4;
        while (blkCnt > 0U)
        {
            f16x8_t vecA;

            vecIn1 = vld1q(pInVec);      
            pInVec += 8;
            vecIn2 = vld1q(pInVec);      
            pInVec += 8;

            tmp = vld1q(pSrcA0Vec);  
            pSrcA0Vec += 16;
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn2, vecA);

            tmp = vld1q(pSrcA1Vec);  
            pSrcA1Vec += 16;
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc1 = vfmaq(acc1, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc1 = vfmaq(acc1, vecIn2, vecA);

            tmp = vld1q(pSrcA2Vec);  
            pSrcA2Vec += 16;
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc2 = vfmaq(acc2, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc2 = vfmaq(acc2, vecIn2, vecA);

            tmp = vld1q(pSrcA3Vec);  
            pSrcA3Vec += 16;
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc3 = vfmaq(acc3, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc3 = vfmaq(acc3, vecIn2, vecA);

            blkCnt--;
        }
        /*
         * tail
         * (will be merged thru tail predication)
         */
        blkCnt = numCols & 0x0F;
        if (blkCnt > 8U)
        {
            mve_pred16_t p0 = vctp16q(blkCnt-8);
            f16x8_t vecA;

            vecIn1 = vld1q(pInVec);      
            pInVec += 8;
            vecIn2 = vldrhq_z_f16(pInVec, p0);

            tmp = vld1q(pSrcA0Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn2, vecA);

            tmp = vld1q(pSrcA1Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc1 = vfmaq(acc1, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc1 = vfmaq(acc1, vecIn2, vecA);

            tmp = vld1q(pSrcA2Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc2 = vfmaq(acc2, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc2 = vfmaq(acc2, vecIn2, vecA);

            tmp = vld1q(pSrcA3Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc3 = vfmaq(acc3, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc3 = vfmaq(acc3, vecIn2, vecA);
        }
        else if (blkCnt > 0U)
        {
            mve_pred16_t p0 = vctp16q(blkCnt);
            f16x8_t vecA;

            vecIn1 = vldrhq_z_f16(pInVec, p0);

            tmp = vld1q(pSrcA0Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn1, vecA);
            
            tmp = vld1q(pSrcA1Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc1 = vfmaq(acc1, vecIn1, vecA);

            tmp = vld1q(pSrcA2Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc2 = vfmaq(acc2, vecIn1, vecA);

            tmp = vld1q(pSrcA3Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc3 = vfmaq(acc3, vecIn1, vecA);
        }
        /*
         * Sum the partial parts
         */
        *px++ = vecAddAcrossF16Mve(acc0);
        *px++ = vecAddAcrossF16Mve(acc1);
        *px++ = vecAddAcrossF16Mve(acc2);
        *px++ = vecAddAcrossF16Mve(acc3);

        pSrcA += numCols * 4;
        /*
         * Decrement the row loop counter
         */
        row -= 4;
    }

    /*
     * compute 2 rows in parrallel
     */
    if (row >= 2)
    {
        float8_t const    *pSrcA0Vec, *pSrcA1Vec;
        float16_t const *pInVec;
        f16x8_t            vecIn1, vecIn2,acc0, acc1;
        float16_t const     *pSrcVecPtr = pSrcVec;
        int8x16_t tmp;

        /*
         * Initialize the pointers to 2 consecutive MatrixA rows
         */
        pInA0 = pSrcA;
        pInA1 = pInA0 + numCols;
        /*
         * Initialize the vector pointer
         */
        pInVec = pSrcVecPtr;
        /*
         * reset accumulators
         */
        acc0 = vdupq_n_f16(0.0f);
        acc1 = vdupq_n_f16(0.0f);
        pSrcA0Vec = pInA0;
        pSrcA1Vec = pInA1;

        blkCnt = numCols >> 4;
        while (blkCnt > 0U)
        {
            f16x8_t vecA;

            vecIn1 = vld1q(pInVec);      
            pInVec += 8;
            vecIn2 = vld1q(pInVec);      
            pInVec += 8;

            tmp = vld1q(pSrcA0Vec);  
            pSrcA0Vec += 16;
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn2, vecA);

            tmp = vld1q(pSrcA1Vec);
            pSrcA1Vec += 16;
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc1 = vfmaq(acc1, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc1 = vfmaq(acc1, vecIn2, vecA);

            blkCnt--;
        }
        /*
         * tail
         * (will be merged thru tail predication)
         */
        blkCnt = numCols & 0x0F;
        if (blkCnt > 8U)
        {
            mve_pred16_t p0 = vctp16q(blkCnt-8);
            f16x8_t vecA;

            vecIn1 = vld1q(pInVec);      
            pInVec += 8;
            vecIn2 = vldrhq_z_f16(pInVec, p0);

            tmp = vld1q(pSrcA0Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn2, vecA);

            tmp = vld1q(pSrcA1Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc1 = vfmaq(acc1, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc1 = vfmaq(acc1, vecIn2, vecA);
        }
        else if (blkCnt > 0U)
        {
            mve_pred16_t p0 = vctp16q(blkCnt);
            f16x8_t vecA;

            vecIn1 = vldrhq_z_f16(pInVec, p0);

            tmp = vld1q(pSrcA0Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn1, vecA);
            
            tmp = vld1q(pSrcA1Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc1 = vfmaq(acc1, vecIn1, vecA);

        }
        /*
         * Sum the partial parts
         */
        *px++ = vecAddAcrossF16Mve(acc0);
        *px++ = vecAddAcrossF16Mve(acc1);

        pSrcA += numCols * 2;
        row -= 2;
    }

    if (row >= 1)
    {
        f16x8_t             vecIn1, vecIn2, acc0;
        float8_t const     *pSrcA0Vec;
        float16_t const *pInVec;
        float16_t const      *pSrcVecPtr = pSrcVec;
        int8x16_t tmp;
        /*
         * Initialize the pointers to last MatrixA row
         */
        pInA0 = pSrcA;
        /*
         * Initialize the vector pointer
         */
        pInVec = pSrcVecPtr;
        /*
         * reset accumulators
         */
        acc0 = vdupq_n_f16(0.0f);

        pSrcA0Vec = pInA0;

        blkCnt = numCols >> 4;
        while (blkCnt > 0U)
        {
            f16x8_t vecA;

            vecIn1 = vld1q(pInVec);      
            pInVec += 8;
            vecIn2 = vld1q(pInVec);      
            pInVec += 8;

            tmp = vld1q(pSrcA0Vec);  
            pSrcA0Vec += 16;
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn2, vecA);

            blkCnt--;
        }
        /*
         * tail
         * (will be merged thru tail predication)
         */
        blkCnt = numCols & 0x0F;
        if (blkCnt > 8U)
        {
           mve_pred16_t p0 = vctp16q(blkCnt-8);
            f16x8_t vecA;

            vecIn1 = vld1q(pInVec);      
            pInVec += 8;
            vecIn2 = vldrhq_z_f16(pInVec, p0);

            tmp = vld1q(pSrcA0Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn1, vecA);
            vecA = vreinterpretq_f16_s16(vshlltq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn2, vecA);
        }
        else if (blkCnt > 0U)
        {
            mve_pred16_t p0 = vctp16q(blkCnt);
            f16x8_t vecA;

            vecIn1 = vldrhq_z_f16(pInVec, p0);

            tmp = vld1q(pSrcA0Vec);
            vecA = vreinterpretq_f16_s16(vshllbq_n_s8(tmp,8));
            acc0 = vfmaq(acc0, vecIn1, vecA);
        }
        /*
         * Sum the partial parts
         */
        *px++ = vecAddAcrossF16Mve(acc0);
    }
}