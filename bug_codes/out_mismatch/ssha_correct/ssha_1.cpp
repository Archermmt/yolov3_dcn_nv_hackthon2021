#include "ssha_1.h"
#include "tensorrt/trans_image.h"
using namespace nvinfer1;
using namespace std;

bool ssha_1::build(TRTUniquePtr<IBuilder>& builder,TRTUniquePtr<INetworkDefinition>& network,TRTUniquePtr<IBuilderConfig>& config,
  ITensor** inputs,ITensor** outputs,int batch_size,DLRLogger& dlr_logger)
{
  // Add Layers
  mWeightsMap = load_weigths("/usr/local/quake/datas/weights/ssha_1.wts");
  // passby data(dlr_input), defined by data;
  ITensor* inputTensors_1[1] = {inputs[0]};
  auto plugin_1=TRANS_IMAGE_Plugin("trans_image",600,600,PluginFormat::kNCHW,ColorOrder::kRGB);
  auto res_1=network->addPluginV2(inputTensors_1,1,plugin_1);
  assert(res_1 && "failed to build image_trans (type:trans_image)");
  res_1->getOutput(0)->setName("image_trans:0");
  auto res_2=network->addScale(*res_1->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_2"],mWeightsMap["scale_2"],mWeightsMap["power_2"]);
  assert(res_2 && "failed to build bn_data (type:scale)");
  res_2->getOutput(0)->setName("bn_data:0");
  auto res_3=network->addConvolution(*res_2->getOutput(0),64,DimsHW{7,7},mWeightsMap["weight_3"],mWeightsMap["bias_3"]);
  res_3->setStride(DimsHW{2,2});
  res_3->setPadding(DimsHW{3,3});
  assert(res_3 && "failed to build conv0 (type:conv2d)");
  res_3->getOutput(0)->setName("conv0:0");
  auto res_4=network->addActivation(*res_3->getOutput(0),ActivationType::kRELU);
  assert(res_4 && "failed to build relu0 (type:relu)");
  res_4->getOutput(0)->setName("relu0:0");
  auto res_5=network->addPooling(*res_4->getOutput(0),PoolingType::kMAX,DimsHW{3,3});
  res_5->setStride(DimsHW{2,2});
  res_5->setPadding(DimsHW{1,1});
  assert(res_5 && "failed to build pooling0 (type:maxpool2d)");
  res_5->getOutput(0)->setName("pooling0:0");
  auto res_6=network->addScale(*res_5->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_6"],mWeightsMap["scale_6"],mWeightsMap["power_6"]);
  assert(res_6 && "failed to build stage1_unit1_bn1 (type:scale)");
  res_6->getOutput(0)->setName("stage1_unit1_bn1:0");
  auto res_7=network->addActivation(*res_6->getOutput(0),ActivationType::kRELU);
  assert(res_7 && "failed to build stage1_unit1_relu1 (type:relu)");
  res_7->getOutput(0)->setName("stage1_unit1_relu1:0");
  auto res_8=network->addConvolution(*res_7->getOutput(0),64,DimsHW{3,3},mWeightsMap["weight_8"],mWeightsMap["bias_8"]);
  res_8->setPadding(DimsHW{1,1});
  assert(res_8 && "failed to build stage1_unit1_conv1 (type:conv2d)");
  res_8->getOutput(0)->setName("stage1_unit1_conv1:0");
  auto res_9=network->addActivation(*res_8->getOutput(0),ActivationType::kRELU);
  assert(res_9 && "failed to build stage1_unit1_relu2 (type:relu)");
  res_9->getOutput(0)->setName("stage1_unit1_relu2:0");
  auto res_10=network->addConvolution(*res_9->getOutput(0),64,DimsHW{3,3},mWeightsMap["weight_10"],mWeightsMap["bias_10"]);
  res_10->setPadding(DimsHW{1,1});
  assert(res_10 && "failed to build stage1_unit1_conv2 (type:conv2d)");
  res_10->getOutput(0)->setName("stage1_unit1_conv2:0");
  auto res_11=network->addConvolution(*res_7->getOutput(0),64,DimsHW{1,1},mWeightsMap["weight_11"],mWeightsMap["bias_11"]);
  assert(res_11 && "failed to build stage1_unit1_sc (type:conv2d)");
  res_11->getOutput(0)->setName("stage1_unit1_sc:0");
  auto res_12=network->addElementWise(*res_10->getOutput(0),*res_11->getOutput(0),ElementWiseOperation::kSUM);
  assert(res_12 && "failed to build _plus0 (type:add)");
  res_12->getOutput(0)->setName("_plus0:0");
  auto res_13=network->addScale(*res_12->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_13"],mWeightsMap["scale_13"],mWeightsMap["power_13"]);
  assert(res_13 && "failed to build stage1_unit2_bn1 (type:scale)");
  res_13->getOutput(0)->setName("stage1_unit2_bn1:0");
  auto res_14=network->addActivation(*res_13->getOutput(0),ActivationType::kRELU);
  assert(res_14 && "failed to build stage1_unit2_relu1 (type:relu)");
  res_14->getOutput(0)->setName("stage1_unit2_relu1:0");
  auto res_15=network->addConvolution(*res_14->getOutput(0),64,DimsHW{3,3},mWeightsMap["weight_15"],mWeightsMap["bias_15"]);
  res_15->setPadding(DimsHW{1,1});
  assert(res_15 && "failed to build stage1_unit2_conv1 (type:conv2d)");
  res_15->getOutput(0)->setName("stage1_unit2_conv1:0");
  auto res_16=network->addActivation(*res_15->getOutput(0),ActivationType::kRELU);
  assert(res_16 && "failed to build stage1_unit2_relu2 (type:relu)");
  res_16->getOutput(0)->setName("stage1_unit2_relu2:0");
  auto res_17=network->addConvolution(*res_16->getOutput(0),64,DimsHW{3,3},mWeightsMap["weight_17"],mWeightsMap["bias_17"]);
  res_17->setPadding(DimsHW{1,1});
  assert(res_17 && "failed to build stage1_unit2_conv2 (type:conv2d)");
  res_17->getOutput(0)->setName("stage1_unit2_conv2:0");
  auto res_18=network->addElementWise(*res_17->getOutput(0),*res_12->getOutput(0),ElementWiseOperation::kSUM);
  assert(res_18 && "failed to build _plus1 (type:add)");
  res_18->getOutput(0)->setName("_plus1:0");
  auto res_19=network->addScale(*res_18->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_19"],mWeightsMap["scale_19"],mWeightsMap["power_19"]);
  assert(res_19 && "failed to build stage2_unit1_bn1 (type:scale)");
  res_19->getOutput(0)->setName("stage2_unit1_bn1:0");
  auto res_20=network->addActivation(*res_19->getOutput(0),ActivationType::kRELU);
  assert(res_20 && "failed to build stage2_unit1_relu1 (type:relu)");
  res_20->getOutput(0)->setName("stage2_unit1_relu1:0");
  auto res_21=network->addConvolution(*res_20->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_21"],mWeightsMap["bias_21"]);
  res_21->setStride(DimsHW{2,2});
  res_21->setPadding(DimsHW{1,1});
  assert(res_21 && "failed to build stage2_unit1_conv1 (type:conv2d)");
  res_21->getOutput(0)->setName("stage2_unit1_conv1:0");
  auto res_22=network->addActivation(*res_21->getOutput(0),ActivationType::kRELU);
  assert(res_22 && "failed to build stage2_unit1_relu2 (type:relu)");
  res_22->getOutput(0)->setName("stage2_unit1_relu2:0");
  auto res_23=network->addConvolution(*res_22->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_23"],mWeightsMap["bias_23"]);
  res_23->setPadding(DimsHW{1,1});
  assert(res_23 && "failed to build stage2_unit1_conv2 (type:conv2d)");
  res_23->getOutput(0)->setName("stage2_unit1_conv2:0");
  auto res_24=network->addConvolution(*res_20->getOutput(0),128,DimsHW{1,1},mWeightsMap["weight_24"],mWeightsMap["bias_24"]);
  res_24->setStride(DimsHW{2,2});
  assert(res_24 && "failed to build stage2_unit1_sc (type:conv2d)");
  res_24->getOutput(0)->setName("stage2_unit1_sc:0");
  auto res_25=network->addElementWise(*res_23->getOutput(0),*res_24->getOutput(0),ElementWiseOperation::kSUM);
  assert(res_25 && "failed to build _plus2 (type:add)");
  res_25->getOutput(0)->setName("_plus2:0");
  auto res_26=network->addScale(*res_25->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_26"],mWeightsMap["scale_26"],mWeightsMap["power_26"]);
  assert(res_26 && "failed to build stage2_unit2_bn1 (type:scale)");
  res_26->getOutput(0)->setName("stage2_unit2_bn1:0");
  auto res_27=network->addActivation(*res_26->getOutput(0),ActivationType::kRELU);
  assert(res_27 && "failed to build stage2_unit2_relu1 (type:relu)");
  res_27->getOutput(0)->setName("stage2_unit2_relu1:0");
  auto res_28=network->addConvolution(*res_27->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_28"],mWeightsMap["bias_28"]);
  res_28->setPadding(DimsHW{1,1});
  assert(res_28 && "failed to build stage2_unit2_conv1 (type:conv2d)");
  res_28->getOutput(0)->setName("stage2_unit2_conv1:0");
  auto res_29=network->addActivation(*res_28->getOutput(0),ActivationType::kRELU);
  assert(res_29 && "failed to build stage2_unit2_relu2 (type:relu)");
  res_29->getOutput(0)->setName("stage2_unit2_relu2:0");
  auto res_30=network->addConvolution(*res_29->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_30"],mWeightsMap["bias_30"]);
  res_30->setPadding(DimsHW{1,1});
  assert(res_30 && "failed to build stage2_unit2_conv2 (type:conv2d)");
  res_30->getOutput(0)->setName("stage2_unit2_conv2:0");
  auto res_31=network->addElementWise(*res_30->getOutput(0),*res_25->getOutput(0),ElementWiseOperation::kSUM);
  assert(res_31 && "failed to build _plus3 (type:add)");
  res_31->getOutput(0)->setName("_plus3:0");
  auto res_32=network->addScale(*res_31->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_32"],mWeightsMap["scale_32"],mWeightsMap["power_32"]);
  assert(res_32 && "failed to build stage3_unit1_bn1 (type:scale)");
  res_32->getOutput(0)->setName("stage3_unit1_bn1:0");
  auto res_33=network->addActivation(*res_32->getOutput(0),ActivationType::kRELU);
  assert(res_33 && "failed to build stage3_unit1_relu1 (type:relu)");
  res_33->getOutput(0)->setName("stage3_unit1_relu1:0");
  auto res_34=network->addConvolution(*res_33->getOutput(0),256,DimsHW{3,3},mWeightsMap["weight_34"],mWeightsMap["bias_34"]);
  res_34->setStride(DimsHW{2,2});
  res_34->setPadding(DimsHW{1,1});
  assert(res_34 && "failed to build stage3_unit1_conv1 (type:conv2d)");
  res_34->getOutput(0)->setName("stage3_unit1_conv1:0");
  auto res_35=network->addActivation(*res_34->getOutput(0),ActivationType::kRELU);
  assert(res_35 && "failed to build stage3_unit1_relu2 (type:relu)");
  res_35->getOutput(0)->setName("stage3_unit1_relu2:0");
  auto res_36=network->addConvolution(*res_35->getOutput(0),256,DimsHW{3,3},mWeightsMap["weight_36"],mWeightsMap["bias_36"]);
  res_36->setPadding(DimsHW{1,1});
  assert(res_36 && "failed to build stage3_unit1_conv2 (type:conv2d)");
  res_36->getOutput(0)->setName("stage3_unit1_conv2:0");
  auto res_37=network->addConvolution(*res_33->getOutput(0),256,DimsHW{1,1},mWeightsMap["weight_37"],mWeightsMap["bias_37"]);
  res_37->setStride(DimsHW{2,2});
  assert(res_37 && "failed to build stage3_unit1_sc (type:conv2d)");
  res_37->getOutput(0)->setName("stage3_unit1_sc:0");
  auto res_38=network->addElementWise(*res_36->getOutput(0),*res_37->getOutput(0),ElementWiseOperation::kSUM);
  assert(res_38 && "failed to build _plus4 (type:add)");
  res_38->getOutput(0)->setName("_plus4:0");
  auto res_39=network->addScale(*res_38->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_39"],mWeightsMap["scale_39"],mWeightsMap["power_39"]);
  assert(res_39 && "failed to build stage3_unit2_bn1 (type:scale)");
  res_39->getOutput(0)->setName("stage3_unit2_bn1:0");
  auto res_40=network->addActivation(*res_39->getOutput(0),ActivationType::kRELU);
  assert(res_40 && "failed to build stage3_unit2_relu1 (type:relu)");
  res_40->getOutput(0)->setName("stage3_unit2_relu1:0");
  auto res_41=network->addConvolution(*res_40->getOutput(0),256,DimsHW{3,3},mWeightsMap["weight_41"],mWeightsMap["bias_41"]);
  res_41->setPadding(DimsHW{1,1});
  assert(res_41 && "failed to build stage3_unit2_conv1 (type:conv2d)");
  res_41->getOutput(0)->setName("stage3_unit2_conv1:0");
  auto res_42=network->addActivation(*res_41->getOutput(0),ActivationType::kRELU);
  assert(res_42 && "failed to build stage3_unit2_relu2 (type:relu)");
  res_42->getOutput(0)->setName("stage3_unit2_relu2:0");
  auto res_43=network->addConvolution(*res_42->getOutput(0),256,DimsHW{3,3},mWeightsMap["weight_43"],mWeightsMap["bias_43"]);
  res_43->setPadding(DimsHW{1,1});
  assert(res_43 && "failed to build stage3_unit2_conv2 (type:conv2d)");
  res_43->getOutput(0)->setName("stage3_unit2_conv2:0");
  auto res_44=network->addElementWise(*res_43->getOutput(0),*res_38->getOutput(0),ElementWiseOperation::kSUM);
  assert(res_44 && "failed to build _plus5 (type:add)");
  res_44->getOutput(0)->setName("_plus5:0");
  auto res_45=network->addScale(*res_44->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_45"],mWeightsMap["scale_45"],mWeightsMap["power_45"]);
  assert(res_45 && "failed to build stage4_unit1_bn1 (type:scale)");
  res_45->getOutput(0)->setName("stage4_unit1_bn1:0");
  auto res_46=network->addActivation(*res_45->getOutput(0),ActivationType::kRELU);
  assert(res_46 && "failed to build stage4_unit1_relu1 (type:relu)");
  res_46->getOutput(0)->setName("stage4_unit1_relu1:0");
  auto res_47=network->addConvolution(*res_46->getOutput(0),512,DimsHW{3,3},mWeightsMap["weight_47"],mWeightsMap["bias_47"]);
  res_47->setStride(DimsHW{2,2});
  res_47->setPadding(DimsHW{1,1});
  assert(res_47 && "failed to build stage4_unit1_conv1 (type:conv2d)");
  res_47->getOutput(0)->setName("stage4_unit1_conv1:0");
  auto res_48=network->addActivation(*res_47->getOutput(0),ActivationType::kRELU);
  assert(res_48 && "failed to build stage4_unit1_relu2 (type:relu)");
  res_48->getOutput(0)->setName("stage4_unit1_relu2:0");
  auto res_49=network->addConvolution(*res_48->getOutput(0),512,DimsHW{3,3},mWeightsMap["weight_49"],mWeightsMap["bias_49"]);
  res_49->setPadding(DimsHW{1,1});
  assert(res_49 && "failed to build stage4_unit1_conv2 (type:conv2d)");
  res_49->getOutput(0)->setName("stage4_unit1_conv2:0");
  auto res_50=network->addConvolution(*res_46->getOutput(0),512,DimsHW{1,1},mWeightsMap["weight_50"],mWeightsMap["bias_50"]);
  res_50->setStride(DimsHW{2,2});
  assert(res_50 && "failed to build stage4_unit1_sc (type:conv2d)");
  res_50->getOutput(0)->setName("stage4_unit1_sc:0");
  auto res_51=network->addElementWise(*res_49->getOutput(0),*res_50->getOutput(0),ElementWiseOperation::kSUM);
  assert(res_51 && "failed to build _plus6 (type:add)");
  res_51->getOutput(0)->setName("_plus6:0");
  auto res_52=network->addScale(*res_51->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_52"],mWeightsMap["scale_52"],mWeightsMap["power_52"]);
  assert(res_52 && "failed to build stage4_unit2_bn1 (type:scale)");
  res_52->getOutput(0)->setName("stage4_unit2_bn1:0");
  auto res_53=network->addActivation(*res_52->getOutput(0),ActivationType::kRELU);
  assert(res_53 && "failed to build stage4_unit2_relu1 (type:relu)");
  res_53->getOutput(0)->setName("stage4_unit2_relu1:0");
  auto res_54=network->addConvolution(*res_53->getOutput(0),512,DimsHW{3,3},mWeightsMap["weight_54"],mWeightsMap["bias_54"]);
  res_54->setPadding(DimsHW{1,1});
  assert(res_54 && "failed to build stage4_unit2_conv1 (type:conv2d)");
  res_54->getOutput(0)->setName("stage4_unit2_conv1:0");
  auto res_55=network->addActivation(*res_54->getOutput(0),ActivationType::kRELU);
  assert(res_55 && "failed to build stage4_unit2_relu2 (type:relu)");
  res_55->getOutput(0)->setName("stage4_unit2_relu2:0");
  auto res_56=network->addConvolution(*res_55->getOutput(0),512,DimsHW{3,3},mWeightsMap["weight_56"],mWeightsMap["bias_56"]);
  res_56->setPadding(DimsHW{1,1});
  assert(res_56 && "failed to build stage4_unit2_conv2 (type:conv2d)");
  res_56->getOutput(0)->setName("stage4_unit2_conv2:0");
  auto res_57=network->addElementWise(*res_56->getOutput(0),*res_51->getOutput(0),ElementWiseOperation::kSUM);
  assert(res_57 && "failed to build _plus7 (type:add)");
  res_57->getOutput(0)->setName("_plus7:0");
  auto res_58=network->addScale(*res_57->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_58"],mWeightsMap["scale_58"],mWeightsMap["power_58"]);
  assert(res_58 && "failed to build bn1 (type:scale)");
  res_58->getOutput(0)->setName("bn1:0");
  auto res_59=network->addActivation(*res_58->getOutput(0),ActivationType::kRELU);
  assert(res_59 && "failed to build relu1 (type:relu)");
  res_59->getOutput(0)->setName("relu1:0");
  auto res_60=network->addConvolution(*res_59->getOutput(0),256,DimsHW{3,3},mWeightsMap["weight_60"],mWeightsMap["bias_60"]);
  res_60->setPadding(DimsHW{1,1});
  assert(res_60 && "failed to build ssh_m3_det_conv1 (type:conv2d)");
  res_60->getOutput(0)->setName("ssh_m3_det_conv1:0");
  auto res_61=network->addConvolution(*res_59->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_61"],mWeightsMap["bias_61"]);
  res_61->setPadding(DimsHW{1,1});
  assert(res_61 && "failed to build ssh_m3_det_context_conv1 (type:conv2d)");
  res_61->getOutput(0)->setName("ssh_m3_det_context_conv1:0");
  auto res_62=network->addActivation(*res_61->getOutput(0),ActivationType::kRELU);
  assert(res_62 && "failed to build ssh_m3_det_context_conv1_relu (type:relu)");
  res_62->getOutput(0)->setName("ssh_m3_det_context_conv1_relu:0");
  auto res_63=network->addConvolution(*res_62->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_63"],mWeightsMap["bias_63"]);
  res_63->setPadding(DimsHW{1,1});
  assert(res_63 && "failed to build ssh_m3_det_context_conv2 (type:conv2d)");
  res_63->getOutput(0)->setName("ssh_m3_det_context_conv2:0");
  auto res_64=network->addConvolution(*res_62->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_64"],mWeightsMap["bias_64"]);
  res_64->setPadding(DimsHW{1,1});
  assert(res_64 && "failed to build ssh_m3_det_context_conv3_1 (type:conv2d)");
  res_64->getOutput(0)->setName("ssh_m3_det_context_conv3_1:0");
  auto res_65=network->addActivation(*res_64->getOutput(0),ActivationType::kRELU);
  assert(res_65 && "failed to build ssh_m3_det_context_conv3_1_relu (type:relu)");
  res_65->getOutput(0)->setName("ssh_m3_det_context_conv3_1_relu:0");
  auto res_66=network->addConvolution(*res_65->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_66"],mWeightsMap["bias_66"]);
  res_66->setPadding(DimsHW{1,1});
  assert(res_66 && "failed to build ssh_m3_det_context_conv3_2 (type:conv2d)");
  res_66->getOutput(0)->setName("ssh_m3_det_context_conv3_2:0");
  ITensor* inputTensors_67[3] = {res_60->getOutput(0),res_63->getOutput(0),res_66->getOutput(0)};
  auto res_67=network->addConcatenation(inputTensors_67,3);
  res_67->setAxis(0);
  assert(res_67 && "failed to build ssh_m3_det_concat (type:concat)");
  res_67->getOutput(0)->setName("ssh_m3_det_concat:0");
  auto res_68=network->addActivation(*res_67->getOutput(0),ActivationType::kRELU);
  assert(res_68 && "failed to build ssh_m3_det_concat_relu (type:relu)");
  res_68->getOutput(0)->setName("ssh_m3_det_concat_relu:0");
  auto res_69=network->addConvolution(*res_68->getOutput(0),8,DimsHW{1,1},mWeightsMap["weight_69"],mWeightsMap["bias_69"]);
  assert(res_69 && "failed to build rpn_bbox_pred_stride32 (type:conv2d)");
  res_69->getOutput(0)->setName("rpn_bbox_pred_stride32:0");
  auto res_70=network->addShuffle(*res_69->getOutput(0));
  Permutation permute_70;
  permute_70.order[0]=1;
  permute_70.order[1]=2;
  permute_70.order[2]=0;
  res_70->setFirstTranspose(permute_70);
  assert(res_70 && "failed to build transpose4 (type:transpose)");
  res_70->getOutput(0)->setName("transpose4:0");
  auto res_71=network->addShuffle(*res_70->getOutput(0));
  res_71->setReshapeDimensions(Dims{2,{722,4},{DimensionType::kCHANNEL,DimensionType::kSPATIAL}});
  assert(res_71 && "failed to build reshape4 (type:reshape)");
  res_71->getOutput(0)->setName("reshape4");
  outputs[0]=res_71->getOutput(0);
  auto res_72=network->addConvolution(*res_68->getOutput(0),4,DimsHW{1,1},mWeightsMap["weight_72"],mWeightsMap["bias_72"]);
  assert(res_72 && "failed to build rpn_cls_score_stride32 (type:conv2d)");
  res_72->getOutput(0)->setName("rpn_cls_score_stride32:0");
  auto res_73=network->addShuffle(*res_72->getOutput(0));
  res_73->setReshapeDimensions(Dims3{2,38,19});
  assert(res_73 && "failed to build rpn_cls_score_reshape_stride32 (type:reshape)");
  res_73->getOutput(0)->setName("rpn_cls_score_reshape_stride32:0");
  auto res_74=network->addSoftMax(*res_73->getOutput(0));
  res_74->setAxes(1);
  assert(res_74 && "failed to build rpn_cls_prob_stride32 (type:softmax)");
  res_74->getOutput(0)->setName("rpn_cls_prob_stride32:0");
  auto res_75=network->addShuffle(*res_74->getOutput(0));
  res_75->setReshapeDimensions(Dims3{4,19,19});
  assert(res_75 && "failed to build rpn_cls_prob_reshape_stride32 (type:reshape)");
  res_75->getOutput(0)->setName("rpn_cls_prob_reshape_stride32:0");
  auto res_76=network->addShuffle(*res_75->getOutput(0));
  Permutation permute_76;
  permute_76.order[0]=1;
  permute_76.order[1]=2;
  permute_76.order[2]=0;
  res_76->setFirstTranspose(permute_76);
  assert(res_76 && "failed to build transpose5 (type:transpose)");
  res_76->getOutput(0)->setName("transpose5:0");
  auto res_77=network->addSlice(*res_76->getOutput(0),Dims3{0,0,2},Dims3{19,19,2},Dims3{1,1,1});
  assert(res_77 && "failed to build slice_axis2 (type:dlr_slice)");
  res_77->getOutput(0)->setName("slice_axis2:0");
  auto res_78=network->addShuffle(*res_77->getOutput(0));
  res_78->setReshapeDimensions(Dims{2,{722,1},{DimensionType::kCHANNEL,DimensionType::kSPATIAL}});
  assert(res_78 && "failed to build reshape5 (type:reshape)");
  res_78->getOutput(0)->setName("reshape5");
  outputs[1]=res_78->getOutput(0);
  auto res_79=network->addScale(*res_44->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_79"],mWeightsMap["scale_79"],mWeightsMap["power_79"]);
  assert(res_79 && "failed to build out_bn4 (type:scale)");
  res_79->getOutput(0)->setName("out_bn4:0");
  auto res_80=network->addActivation(*res_79->getOutput(0),ActivationType::kRELU);
  assert(res_80 && "failed to build out_relu4 (type:relu)");
  res_80->getOutput(0)->setName("out_relu4:0");
  auto res_81=network->addConvolution(*res_80->getOutput(0),256,DimsHW{3,3},mWeightsMap["weight_81"],mWeightsMap["bias_81"]);
  res_81->setPadding(DimsHW{1,1});
  assert(res_81 && "failed to build ssh_m2_det_conv1 (type:conv2d)");
  res_81->getOutput(0)->setName("ssh_m2_det_conv1:0");
  auto res_82=network->addConvolution(*res_80->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_82"],mWeightsMap["bias_82"]);
  res_82->setPadding(DimsHW{1,1});
  assert(res_82 && "failed to build ssh_m2_det_context_conv1 (type:conv2d)");
  res_82->getOutput(0)->setName("ssh_m2_det_context_conv1:0");
  auto res_83=network->addActivation(*res_82->getOutput(0),ActivationType::kRELU);
  assert(res_83 && "failed to build ssh_m2_det_context_conv1_relu (type:relu)");
  res_83->getOutput(0)->setName("ssh_m2_det_context_conv1_relu:0");
  auto res_84=network->addConvolution(*res_83->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_84"],mWeightsMap["bias_84"]);
  res_84->setPadding(DimsHW{1,1});
  assert(res_84 && "failed to build ssh_m2_det_context_conv2 (type:conv2d)");
  res_84->getOutput(0)->setName("ssh_m2_det_context_conv2:0");
  auto res_85=network->addConvolution(*res_83->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_85"],mWeightsMap["bias_85"]);
  res_85->setPadding(DimsHW{1,1});
  assert(res_85 && "failed to build ssh_m2_det_context_conv3_1 (type:conv2d)");
  res_85->getOutput(0)->setName("ssh_m2_det_context_conv3_1:0");
  auto res_86=network->addActivation(*res_85->getOutput(0),ActivationType::kRELU);
  assert(res_86 && "failed to build ssh_m2_det_context_conv3_1_relu (type:relu)");
  res_86->getOutput(0)->setName("ssh_m2_det_context_conv3_1_relu:0");
  auto res_87=network->addConvolution(*res_86->getOutput(0),128,DimsHW{3,3},mWeightsMap["weight_87"],mWeightsMap["bias_87"]);
  res_87->setPadding(DimsHW{1,1});
  assert(res_87 && "failed to build ssh_m2_det_context_conv3_2 (type:conv2d)");
  res_87->getOutput(0)->setName("ssh_m2_det_context_conv3_2:0");
  ITensor* inputTensors_88[3] = {res_81->getOutput(0),res_84->getOutput(0),res_87->getOutput(0)};
  auto res_88=network->addConcatenation(inputTensors_88,3);
  res_88->setAxis(0);
  assert(res_88 && "failed to build ssh_m2_det_concat (type:concat)");
  res_88->getOutput(0)->setName("ssh_m2_det_concat:0");
  auto res_89=network->addActivation(*res_88->getOutput(0),ActivationType::kRELU);
  assert(res_89 && "failed to build ssh_m2_det_concat_relu (type:relu)");
  res_89->getOutput(0)->setName("ssh_m2_det_concat_relu:0");
  auto res_90=network->addConvolution(*res_89->getOutput(0),8,DimsHW{1,1},mWeightsMap["weight_90"],mWeightsMap["bias_90"]);
  assert(res_90 && "failed to build rpn_bbox_pred_stride16 (type:conv2d)");
  res_90->getOutput(0)->setName("rpn_bbox_pred_stride16:0");
  auto res_91=network->addShuffle(*res_90->getOutput(0));
  Permutation permute_91;
  permute_91.order[0]=1;
  permute_91.order[1]=2;
  permute_91.order[2]=0;
  res_91->setFirstTranspose(permute_91);
  assert(res_91 && "failed to build transpose2 (type:transpose)");
  res_91->getOutput(0)->setName("transpose2:0");
  auto res_92=network->addShuffle(*res_91->getOutput(0));
  res_92->setReshapeDimensions(Dims{2,{2888,4},{DimensionType::kCHANNEL,DimensionType::kSPATIAL}});
  assert(res_92 && "failed to build reshape2 (type:reshape)");
  res_92->getOutput(0)->setName("reshape2");
  outputs[2]=res_92->getOutput(0);
  auto res_93=network->addConvolution(*res_89->getOutput(0),4,DimsHW{1,1},mWeightsMap["weight_93"],mWeightsMap["bias_93"]);
  assert(res_93 && "failed to build rpn_cls_score_stride16 (type:conv2d)");
  res_93->getOutput(0)->setName("rpn_cls_score_stride16:0");
  auto res_94=network->addShuffle(*res_93->getOutput(0));
  res_94->setReshapeDimensions(Dims3{2,76,38});
  assert(res_94 && "failed to build rpn_cls_score_reshape_stride16 (type:reshape)");
  res_94->getOutput(0)->setName("rpn_cls_score_reshape_stride16:0");
  auto res_95=network->addSoftMax(*res_94->getOutput(0));
  res_95->setAxes(1);
  assert(res_95 && "failed to build rpn_cls_prob_stride16 (type:softmax)");
  res_95->getOutput(0)->setName("rpn_cls_prob_stride16:0");
  auto res_96=network->addShuffle(*res_95->getOutput(0));
  res_96->setReshapeDimensions(Dims3{4,38,38});
  assert(res_96 && "failed to build rpn_cls_prob_reshape_stride16 (type:reshape)");
  res_96->getOutput(0)->setName("rpn_cls_prob_reshape_stride16:0");
  auto res_97=network->addShuffle(*res_96->getOutput(0));
  Permutation permute_97;
  permute_97.order[0]=1;
  permute_97.order[1]=2;
  permute_97.order[2]=0;
  res_97->setFirstTranspose(permute_97);
  assert(res_97 && "failed to build transpose3 (type:transpose)");
  res_97->getOutput(0)->setName("transpose3:0");
  auto res_98=network->addSlice(*res_97->getOutput(0),Dims3{0,0,2},Dims3{38,38,2},Dims3{1,1,1});
  assert(res_98 && "failed to build slice_axis1 (type:dlr_slice)");
  res_98->getOutput(0)->setName("slice_axis1:0");
  auto res_99=network->addShuffle(*res_98->getOutput(0));
  res_99->setReshapeDimensions(Dims{2,{2888,1},{DimensionType::kCHANNEL,DimensionType::kSPATIAL}});
  assert(res_99 && "failed to build reshape3 (type:reshape)");
  res_99->getOutput(0)->setName("reshape3");
  outputs[3]=res_99->getOutput(0);
  auto res_100=network->addConvolution(*res_80->getOutput(0),128,DimsHW{1,1},mWeightsMap["weight_100"],mWeightsMap["bias_100"]);
  assert(res_100 && "failed to build ssh_m2_red_conv (type:conv2d)");
  res_100->getOutput(0)->setName("ssh_m2_red_conv:0");
  auto res_101=network->addActivation(*res_100->getOutput(0),ActivationType::kRELU);
  assert(res_101 && "failed to build ssh_m2_red_conv_relu (type:relu)");
  res_101->getOutput(0)->setName("ssh_m2_red_conv_relu:0");
  auto res_102=network->addDeconvolution(*res_101->getOutput(0),128,DimsHW{4,4},mWeightsMap["weight_102"],mWeightsMap["bias_102"]);
  res_102->setStride(DimsHW{2,2});
  res_102->setNbGroups(128);
  res_102->setPadding(DimsHW{1,1});
  assert(res_102 && "failed to build ssh_m2_red_upsampling (type:deconv2d)");
  res_102->getOutput(0)->setName("ssh_m2_red_upsampling:0");
  auto res_103=network->addSlice(*res_102->getOutput(0),Dims3{0,0,0},Dims3{128,75,75},Dims3{1,1,1});
  assert(res_103 && "failed to build slice0 (type:dlr_slice)");
  res_103->getOutput(0)->setName("slice0");
  outputs[6]=res_103->getOutput(0);
  auto res_104=network->addScale(*res_31->getOutput(0),ScaleMode::kCHANNEL,mWeightsMap["offset_104"],mWeightsMap["scale_104"],mWeightsMap["power_104"]);
  assert(res_104 && "failed to build out_bn3 (type:scale)");
  res_104->getOutput(0)->setName("out_bn3:0");
  auto res_105=network->addActivation(*res_104->getOutput(0),ActivationType::kRELU);
  assert(res_105 && "failed to build out_relu3 (type:relu)");
  res_105->getOutput(0)->setName("out_relu3:0");
  auto res_106=network->addConvolution(*res_105->getOutput(0),128,DimsHW{1,1},mWeightsMap["weight_106"],mWeightsMap["bias_106"]);
  assert(res_106 && "failed to build ssh_m1_red_conv (type:conv2d)");
  res_106->getOutput(0)->setName("ssh_m1_red_conv:0");
  auto res_107=network->addActivation(*res_106->getOutput(0),ActivationType::kRELU);
  assert(res_107 && "failed to build ssh_m1_red_conv_relu (type:relu)");
  res_107->getOutput(0)->setName("ssh_m1_red_conv_relu");
  outputs[5]=res_107->getOutput(0);
  auto res_108=network->addElementWise(*res_107->getOutput(0),*res_103->getOutput(0),ElementWiseOperation::kSUM);
  assert(res_108 && "failed to build _plus8 (type:add)");
  res_108->getOutput(0)->setName("_plus8");
  outputs[4]=res_108->getOutput(0);
  // Set configs
  builder->setMaxBatchSize(batch_size);
  config->setMaxWorkspaceSize(91 << 20);
  return true;
}

bool ssha_1::clean_up(){
  return clean_weights(mWeightsMap);
}
