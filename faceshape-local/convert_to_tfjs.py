"""แปลงโมเดล Keras (.h5) เป็น TensorFlow.js บน Windows

ใช้แทนคำสั่ง `tensorflowjs_converter` เพราะบน Windows มีปัญหา 3 อย่าง:
  1. Smart App Control บล็อกไฟล์ tensorflowjs_converter.exe
  2. tensorflowjs import `tensorflow_decision_forests` ซึ่งไม่มีเวอร์ชัน Windows
  3. tensorflowjs import `jax` ซึ่ง DLL ของมันโดน Smart App Control บล็อก

การแปลงไฟล์ Keras .h5 ไม่ได้ใช้สองแพ็กเกจนั้นเลย สคริปต์นี้จึงใส่โมดูลว่างแทนไว้ก่อน import

วิธีใช้:  python convert_to_tfjs.py <model.h5> <output_dir>
"""
import sys
import types


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


_stub("tensorflow_decision_forests")
_jax = _stub("jax")
_jax.experimental = _stub("jax.experimental")
_jax.experimental.jax2tf = _stub("jax.experimental.jax2tf")

from tensorflowjs.converters.converter import convert  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python convert_to_tfjs.py <model.h5> <output_dir>")
    convert(["--input_format", "keras", sys.argv[1], sys.argv[2]])
