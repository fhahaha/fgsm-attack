import layers as l
import utils


def SimpleNet1(x, input_shape, neurons=1024, n_classes=10, non_linearity='relu', create_summaries=True):
    h = x
    h, output_shape = l.flatten(input_shape, h)
    h, output_shape = l.linear(output_shape, neurons, h, name='linear1')
    if create_summaries:
        utils.variable_summaries(h, name='linear-comb-hidden-layer')

    h = l.non_linearity(h, name=non_linearity)
    if create_summaries:
        utils.variable_summaries(h, name='activation-hidden-layer')

    logits, output_shape = l.linear(output_shape, n_classes, h, name='output')
    if create_summaries:
        utils.variable_summaries(logits, name='unscaled-logits-output-layer')

    return logits


def ConvNet(x, input_shape, filters_out=64, n_classes=10, non_linearity='relu'):
    # Basic CNN from Cleverhans MNIST tutorial:
    # https://github.com/mmarius/cleverhans/blob/master/cleverhans_tutorials/tutorial_models.py#L155
    h = x
    input_shape = list(input_shape)
    h, output_shape = l.conv2d(h, kernel_size=8, stride=2,
                               filters_in=input_shape[-1],
                               filters_out=filters_out,
                               padding='SAME',
                               name='conv1')
    h = l.non_linearity(h, name=non_linearity)
    h, output_shape = l.conv2d(h, kernel_size=6, stride=2,
                               filters_in=output_shape[-1],
                               filters_out=filters_out * 2,
                               padding='VALID',
                               name='conv2')
    h = l.non_linearity(h, name=non_linearity)
    h, output_shape = l.conv2d(h, kernel_size=5, stride=1,
                               filters_in=output_shape[-1],
                               filters_out=filters_out * 2,
                               padding='VALID',
                               name='conv3')
    h = l.non_linearity(h, name=non_linearity)

    h, output_shape = l.flatten(input_shape=output_shape, x=h)
    logits, output_shape = l.linear(input_shape=output_shape, n_hidden=n_classes, x=h, name='output-layer')
    utils.variable_summaries(logits, name='unscaled-logits-output-layer')

    return logits
