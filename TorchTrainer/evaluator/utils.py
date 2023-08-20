
from typing import Any, Dict


def get_metric_value(indicator: str, metrics: Dict) -> Any:
    """获取指定 metric 的度量值, metric 可以是 metric 名称或具有 evaluator 前缀的完整名称.
    Args:
        indicator (str): 指标标识符, 可以是 metric 名称 ('AP') 或是具有前缀的完整名称 ('COCO/AP')
        metrics (dict): evaluator 输出的评估结果

    Returns:
        Any: 指定指标的度量值
    """

    if '/' in indicator:
        # The indicator is a full name
        if indicator in metrics:
            return metrics[indicator]
        else:
            raise ValueError(
                f'The indicator "{indicator}" can not match any metric in '
                f'{list(metrics.keys())}')
    else:
        # The indicator is metric name without prefix
        matched = [k for k in metrics.keys() if k.split('/')[-1] == indicator]

        if not matched:
            raise ValueError(
                f'The indicator {indicator} can not match any metric in '
                f'{list(metrics.keys())}')
        elif len(matched) > 1:
            raise ValueError(f'The indicator "{indicator}" matches multiple '
                             f'metrics {matched}')
        else:
            return metrics[matched[0]]
