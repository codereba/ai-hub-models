[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [dla102x: Imagenet classifier and general purpose backbone](https://aihub.qualcomm.com/models/dla102x)

dla is a machine learning model that can classify images from the Imagenet dataset. It is a architecture which can be used as backbone.

This is based on the implementation of dla102x found [here](https://huggingface.co/timm/dla102x.in1k). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/dla102x).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[dla102x]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.dla102x.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.dla102x.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.


## License
* The license for the original implementation of dla102x can be found
  [here](hhttps://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/bsd-3-clause.md).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Deep Layer Aggregation](https://arxiv.org/abs/1707.06484)
* [Source Model Implementation](https://huggingface.co/timm/dla102x.in1k)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).

