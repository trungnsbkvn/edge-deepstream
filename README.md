# EdgeDeepStream Dependencies for Jetson

This repository provides pre-built runtime dependencies for the [EdgeDeepStream](https://github.com/trungnsbkvn/edge-deepstream) application on NVIDIA Jetson platforms (aarch64/arm64). These dependencies are essential for running EdgeDeepStream, a framework for real-time deep learning inference on edge devices.

## Release Information

**Version**: Dependencies v1.0  
**Release Date**: October 15, 2025  
**GitHub Release**: [deps-v1.0](https://github.com/trungnsbkvn/edge-deepstream/releases/tag/deps-v1.0)  
**Commit**: `b7eeff9`

## Included Packages

The following libraries are included in this release:

- **Paho MQTT C v1.3.15**: MQTT client library for lightweight messaging.  
  *License*: Eclipse Public License 2.0
- **Paho MQTT C++ v1.5.4**: C++ wrapper for the Paho MQTT C library.  
  *License*: Eclipse Public License 2.0
- **FAISS v1.7.4**: Library for efficient vector similarity search (CPU-only).  
  *License*: MIT License

## Supported Platforms

- **Target**: NVIDIA Jetson (aarch64/arm64)
- **Operating System**: Ubuntu 20.04+ (tested on 20.04 and 22.04)
- **Architecture**: ARM 64-bit

## Installation

### Automatic Installation
The dependencies are automatically downloaded and installed by running the `install_dependencies.sh` script provided in the [EdgeDeepStream repository](https://github.com/trungnsbkvn/edge-deepstream).

```bash
./install_dependencies.sh
```

### Manual Installation
For manual installation, follow these steps:

1. Download the desired package from the [release page](https://github.com/trungnsbkvn/edge-deepstream/releases/tag/deps-v1.0).
2. Extract the package:
   ```bash
   tar -xzf libpaho-mqtt-c-runtime-1.3.15-jetson-arm64.tar.gz
   ```
3. Navigate to the extracted directory:
   ```bash
   cd libpaho-mqtt-c-runtime-1.3.15-jetson-arm64
   ```
4. Run the installation script with sudo privileges:
   ```bash
   sudo ./install.sh
   ```

Repeat these steps for each package as needed.

## Verifying Package Integrity

To ensure the downloaded packages are not corrupted, verify their integrity using the provided SHA256 checksums:

```bash
sha256sum -c SHA256SUMS
```

The `SHA256SUMS` file is included in the release assets.

## Usage

These dependencies are required for running the EdgeDeepStream application on Jetson devices. Ensure all packages are installed before building or running the application. Refer to the [EdgeDeepStream repository](https://github.com/trungnsbkvn/edge-deepstream) for further instructions on setting up and using the application.

## License

The included packages are distributed under their respective licenses:
- Paho MQTT C: Eclipse Public License 2.0
- Paho MQTT C++: Eclipse Public License 2.0
- FAISS: MIT License

## Contributing

Contributions to improve this dependency package or the EdgeDeepStream project are welcome! Please submit issues or pull requests to the [main repository](https://github.com/trungnsbkvn/edge-deepstream).

## Contact

For questions or support, please open an issue on the [GitHub repository](https://github.com/trungnsbkvn/edge-deepstream) or contact the maintainer at [trungnsbkvn](https://github.com/trungnsbkvn).

---

© 2025 DN, Inc.
