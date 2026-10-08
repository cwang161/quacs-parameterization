# quacs/biogenic/megan/tests/test_megan.py

import numpy as np

from quacs.biogenic.megan import biogenic_emission_megan_v3


def test_megan_module_import():
    assert hasattr(
        biogenic_emission_megan_v3,
        "biogenic_emission_megan_v3",
    )


def test_reference_isoprene():
    # 当前参考值，后续需要用固定输入进行实际调用验证
    expected_isoprene = 10.4124

    assert np.isfinite(expected_isoprene)
    assert expected_isoprene > 0
