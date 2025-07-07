# 📚 SHCI Documentation

This documentation provides comprehensive guides for using and developing with the SHCI (Semistochastic Heat Bath Configuration Interaction) quantum chemistry code.

## 📖 Documentation Overview

### For Users
- **[User Manual](docs/User_Manual.md)** - Complete guide for running SHCI calculations
- **[Configuration Reference](docs/Configuration_Reference.md)** - All configuration parameters and options

### For Developers  
- **[API Documentation](docs/API_Documentation.md)** - Complete API reference for all classes and functions
- **[HTML Documentation](docs/doxygen/html/index.html)** - Interactive API documentation (generated with Doxygen)

## 🚀 Quick Start

1. **Installation**: See [User Manual - Installation](docs/User_Manual.md#installation)
2. **First Calculation**: See [User Manual - Quick Start](docs/User_Manual.md#quick-start)
3. **Configuration**: See [Configuration Reference](docs/Configuration_Reference.md)

## 📁 Documentation Structure

```
docs/
├── User_Manual.md              # Complete user guide (60+ pages)
├── Configuration_Reference.md   # Parameter documentation (60+ pages)  
├── API_Documentation.md        # Developer API reference (50+ pages)
└── doxygen/
    └── html/                   # Interactive HTML documentation
        ├── index.html          # Main documentation page
        ├── annotated.html      # Class index
        └── ...                 # Complete API reference
```

## 🎯 Documentation Features

- **Comprehensive Coverage**: 170+ pages of documentation
- **Real Examples**: H₂O, C₆H₆, Hubbard model calculations
- **Step-by-Step Tutorials**: From basic to advanced usage
- **Complete API Reference**: All classes, functions, and parameters documented
- **Interactive HTML**: Searchable, linked documentation with class diagrams
- **Best Practices**: Performance optimization and troubleshooting guides

## 🔧 For Developers

The API documentation includes:
- Complete class hierarchy and relationships
- Function signatures and parameter descriptions  
- Code examples and usage patterns
- Cross-referenced links between related components
- Detailed algorithm descriptions

## 📋 Documentation Generation

To regenerate the HTML documentation:
```bash
doxygen Doxyfile
```

This will update the interactive HTML documentation in `docs/doxygen/html/`.

---

For questions or contributions to the documentation, please refer to the [User Manual](docs/User_Manual.md) or [API Documentation](docs/API_Documentation.md).