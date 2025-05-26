import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(os.path.dirname(__file__)).parent.parent

# 核心模块导出
__all__ = [
    'VisionTransformer',
    'load_data',
    'create_test_loader',
    'config'
]

# 统一配置
config = {
    'CHECKPOINT_DIR': ROOT_DIR / 'output' / 'cv_checkpoints',
    'RESULTS_DIR': ROOT_DIR / 'output' / 'result',
    'DATA_DIR': ROOT_DIR / 'data',
    'DATA_DIR_PROCESSED': ROOT_DIR / 'data' / 'processed_data'
}

# 延迟导入防止循环依赖
from .model import VisionTransformer  # noqa: E402
from .data import (  # noqa: E402
    load_data,
    create_test_loader
)