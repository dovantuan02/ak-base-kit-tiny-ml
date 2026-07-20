#/bin/sh
export GCC_PATH=/opt/gcc-arm-none-eabi-10.3-2021.10
export PROGRAMER_PATH=~/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin

git submodule update --init --recursive

apply_patch() {
    ROOT_DIR=$(git rev-parse --show-toplevel)
    cd "$ROOT_DIR/application/sources/libraries/$1" || exit 1
    PATCH="$ROOT_DIR/application/sources/libraries/.patchs/$2"
    if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
        echo "Patch already applied"
    elif git apply --check "$PATCH" >/dev/null 2>&1; then
        git apply "$PATCH"
    else
        echo "Patch cannot be applied"
    fi

    cd - >/dev/null || exit 1
}
apply_patch SparkFun_ICM-20948_ArduinoLibrary 01-fix-serial.patch
apply_patch SparkFun_ICM-20948_ArduinoLibrary 02-enable-dmp.patch
apply_patch SparkFun_ICM-20948_ArduinoLibrary 03-undef-for-arm.path
