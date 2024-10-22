#ifndef _MPS3_AUDIO_AN547_H_
#define _MPS3_AUDIO_AN547_H_ 



#ifdef __cplusplus
extern "C" {
#endif
/**
 * \brief CS42L52 Audio Codec error enumeration types
 */
enum audio_codec_mps3_error_t {
    AUDIO_CODEC_MPS3_ERR_NONE = 0,      /*!< No error */
};
/**
 * \brief Initializes Audio Codec
 *
 * \return Returns error code as specified in \ref audio_codec_mps3_error_t
 */
enum audio_codec_mps3_error_t audio_codec_mps3_init(void);

extern void mps3_audio_init(int freq);
extern void mps3_audio_stop();

#ifdef __cplusplus
}
#endif

#endif /* __AUDIO_CODEC_MPS3_H__ */