from nemo.collections.asr.models import EncDecHybridRNNTCTCModel

MODEL_PATH = "resources/model.nemo"


def main():
    model = EncDecHybridRNNTCTCModel.restore_from(restore_path=MODEL_PATH).cuda()
    model.change_decoding_strategy(decoder_type="ctc")
    # for english 160ms model
    # model.encoder.set_default_att_context_size([70,1])
    model.set_export_config({"cache_support": True})
    model.eval()
    model.encoder.export(
        "resources/encoder.onnx",
    )
    model.ctc_decoder.export(
        "resources/decoder.onnx",
    )


if __name__ == "__main__":
    main()
