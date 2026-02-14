# Feature Research

**Domain:** Production-ready scientific Python reconstruction library
**Researched:** 2026-02-14
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| PyPI packaging with pip install | Standard for Python libraries | LOW | Already have pyproject.toml, need to publish |
| Comprehensive API documentation | Users need to understand the library | MEDIUM | Sphinx + ReadTheDocs standard; NumPy-style docstrings |
| Installation instructions | Users need to install dependencies | LOW | Especially critical for AquaCal local dependency |
| Basic usage examples | Users need quickstart guidance | LOW | README examples, tutorial notebook |
| CLI help messages | Users need to discover CLI options | LOW | argparse/click/typer with --help |
| Error messages with context | Users need to debug failures | MEDIUM | Structured errors, not bare exceptions |
| License file | Legal requirement for distribution | LOW | Already MIT in pyproject.toml |
| README with project description | First thing users see | LOW | Need to expand current README |
| Versioning and changelog | Users track changes between releases | LOW | Semantic versioning + CHANGELOG.md |
| Unit tests with CI | Users trust tested code | MEDIUM | Have pytest, need GitHub Actions |
| Type hints | Modern Python expectation | MEDIUM | MyPy already in dev deps |
| Code formatting standard | Collaboration requires consistency | LOW | Black already used |
| Python 3.10+ support | Current Python versions | LOW | Already specified in pyproject.toml |
| Cross-platform support | Works on Linux, macOS, Windows | MEDIUM | PyTorch + Open3D should handle this |
| Device flexibility (CPU/GPU) | Users have different hardware | LOW | Already have device config |
| Standard data formats | Import/export compatibility | MEDIUM | PLY, OBJ for meshes; standard depth map formats |
| Progress indicators | Long operations need feedback | LOW | tqdm already in dependencies |
| Logging support | Users need runtime diagnostics | LOW | Python logging module integration |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Refractive geometry handling | Unique capability for underwater reconstruction | HIGH | Core competency, already implemented |
| End-to-end pipeline from video | Simplifies workflow vs manual steps | MEDIUM | Already have pipeline.py |
| Built-in benchmarking tools | Research reproducibility and evaluation | MEDIUM | Already have benchmark module |
| Multi-camera time-series | Temporal reconstruction capability | MEDIUM | Already supported via VideoSet integration |
| Config-driven workflows | Reproducible experiments | LOW | Already have YAML config |
| Visualization tools | Immediate feedback on results | MEDIUM | Already have visualization module |
| Multiple feature extractors | Flexibility for different scenarios | MEDIUM | Already support LightGlue, RoMa |
| Evaluation metrics suite | Quantitative quality assessment | MEDIUM | Already have evaluation module |
| Interactive parameter tuning | Rapid experimentation | HIGH | Not implemented; valuable for research |
| Pre-computed example datasets | Users can test immediately | LOW | Package sample data or download scripts |
| Integration with AquaCal | Seamless calibration-to-reconstruction | MEDIUM | Already integrated |
| Geometric consistency filtering | Higher quality reconstructions | MEDIUM | Already implemented |
| GPU acceleration support | Faster processing for large datasets | MEDIUM | Already have CUDA support |
| Batch processing capabilities | Process multiple frames efficiently | LOW | CLI should support frame ranges |
| Export to standard 3D formats | Integration with external tools | LOW | Open3D handles PLY/OBJ export |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| GUI application | "Easier for non-programmers" | Huge maintenance burden; diverges from scientific reproducibility; platform-specific issues | Jupyter notebooks + good CLI + visualization outputs |
| Real-time reconstruction | "Live feedback during capture" | Complexity explosion; GPU requirements; domain mismatch (post-processing is standard) | Fast batch processing with progress bars |
| All-in-one calibration + reconstruction | "Single tool for everything" | Violates separation of concerns; AquaCal is already mature | Clear documentation on AquaCal integration |
| Automatic parameter tuning | "Don't want to think about parameters" | Domain-specific knowledge required; no one-size-fits-all | Good defaults + parameter sweep tools + documentation |
| Built-in video capture | "Complete end-to-end solution" | Hardware-specific drivers; scope creep | Focus on processing; users handle capture |
| Database backend | "Organize experiments" | Complexity vs value for target users | Filesystem-based organization + naming conventions |
| Web-based dashboard | "Modern interface" | Server complexity; authentication; deployment overhead | Static HTML reports + Jupyter notebooks |
| Plugin system | "Extensibility" | Premature architecture; maintenance burden | Clean Python API is already extensible |
| Mobile app | "Convenient access" | Wrong platform for computational task | Web visualization of exported results if needed |
| Multi-language support (i18n) | "Broader audience" | Scientific users expect English; maintenance overhead | English documentation with clear terminology |

## Feature Dependencies

```
PyPI Packaging
    └──requires──> Comprehensive API Documentation
    └──requires──> Installation Instructions
    └──requires──> README

Comprehensive API Documentation
    └──requires──> Type Hints
    └──requires──> Docstrings (NumPy style)

CI/CD Pipeline
    └──requires──> Unit Tests
    └──requires──> PyPI Packaging
    └──requires──> Code Formatting

Benchmarking Tools
    └──requires──> Evaluation Metrics Suite
    └──requires──> Visualization Tools

Interactive Parameter Tuning
    └──requires──> Visualization Tools
    └──requires──> Config-driven Workflows
    └──conflicts──> Batch Processing (different use cases)

Pre-computed Example Datasets
    └──requires──> Standard Data Formats
    └──enhances──> Basic Usage Examples

GPU Acceleration
    └──requires──> Device Flexibility
    └──requires──> Benchmarking Tools (to measure speedup)
```

### Dependency Notes

- **PyPI Packaging requires documentation:** Users won't adopt library without docs
- **CI/CD requires tests:** Can't automate quality checks without test suite
- **Benchmarking requires metrics:** Need quantitative evaluation capability
- **Interactive tuning conflicts with batch processing:** Different interaction models; focus on batch + visualization
- **Example datasets enhance tutorials:** Concrete examples more valuable than abstract instructions
- **GPU acceleration requires benchmarking:** Need to validate performance claims

## MVP Definition

### Launch With (v0.1.0 - Current Alpha)

Minimum viable product for research community validation.

- [x] Core reconstruction pipeline (already working)
- [x] Basic CLI (init, run, export-refs, benchmark implemented)
- [x] YAML config system (already working)
- [x] Evaluation metrics (already implemented)
- [ ] PyPI packaging (ready to publish)
- [ ] Basic documentation (README, installation guide)
- [ ] GitHub Actions CI (run tests on push)
- [ ] Example dataset or tutorial data
- [ ] Type hints throughout codebase
- [ ] CHANGELOG.md for version tracking

### Add After Validation (v0.2.0 - Beta)

Features to add once core is validated by early users.

- [ ] Comprehensive Sphinx documentation - After early users provide feedback on pain points
- [ ] ReadTheDocs hosting - Once documentation exists
- [ ] Jupyter notebook tutorials - After identifying common use cases from users
- [ ] Progress bars with logging integration - When long-running operations are identified
- [ ] Better error messages with diagnostics - Based on user bug reports
- [ ] CLI improvements (frame ranges, verbosity control) - After usage patterns emerge
- [ ] Pre-trained model weights distribution - If applicable for feature extractors
- [ ] Performance benchmarks - After optimization priorities clear
- [ ] Docker container for reproducibility - When users request it

### Future Consideration (v1.0+)

Features to defer until product-market fit is established.

- [ ] Interactive visualization tools - High complexity, unclear value vs static viz
- [ ] Advanced config validation with schema - After config patterns stabilize
- [ ] Plugin architecture for custom extractors - Wait for external contributor demand
- [ ] Automated performance profiling - After identifying actual bottlenecks
- [ ] Multi-format dataset loaders - When users request specific formats
- [ ] Cloud processing integration - If large-scale processing becomes common

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| PyPI packaging | HIGH | LOW | P1 |
| Basic documentation (README + install) | HIGH | LOW | P1 |
| GitHub Actions CI | HIGH | LOW | P1 |
| Type hints throughout | MEDIUM | MEDIUM | P1 |
| CHANGELOG.md | MEDIUM | LOW | P1 |
| Example dataset/tutorial | HIGH | MEDIUM | P1 |
| Sphinx API documentation | HIGH | MEDIUM | P2 |
| ReadTheDocs hosting | HIGH | LOW | P2 |
| Jupyter tutorials | HIGH | MEDIUM | P2 |
| Progress bars + logging | MEDIUM | LOW | P2 |
| Better error messages | MEDIUM | MEDIUM | P2 |
| CLI improvements | MEDIUM | LOW | P2 |
| Docker container | MEDIUM | MEDIUM | P2 |
| Performance benchmarks | LOW | MEDIUM | P3 |
| Interactive visualization | LOW | HIGH | P3 |
| Advanced config validation | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for v0.1.0 public release (production-ready baseline)
- P2: Should have for v0.2.0 beta (community adoption features)
- P3: Nice to have for v1.0+ (maturity features)

## Competitor Feature Analysis

| Feature | Open3D | PyTorch3D | COLMAP (pycolmap) | AquaMVS Approach |
|---------|--------|-----------|-------------------|------------------|
| PyPI distribution | Yes (pip) | Yes (conda/pip) | Yes (pip) | **Needed** - not yet published |
| Comprehensive docs | Yes (Sphinx) | Yes (Sphinx) | Yes (Sphinx) | **Needed** - minimal README only |
| API documentation | Yes (auto-generated) | Yes (auto-generated) | Yes (auto-generated) | **Needed** - docstrings exist but not published |
| CLI tools | Yes (Open3D-Viewer) | No (Python API only) | Yes (colmap CLI + Python) | **Partial** - have CLI but needs polish |
| Example datasets | Yes (download scripts) | Yes (tutorials with data) | No (users provide) | **Needed** - would lower barrier to entry |
| Jupyter tutorials | Yes (tutorial notebooks) | Yes (extensive) | Limited | **Needed** - key for research adoption |
| GPU acceleration | Yes (documented) | Yes (PyTorch native) | Yes (if compiled with CUDA) | **Have** - via PyTorch + config |
| Pre-trained models | No (geometry lib) | Yes (model zoo) | No (SfM lib) | **N/A** - geometry-based not learning-based |
| Benchmarking tools | Limited | Yes (TorchBench integration) | No | **Have** - custom benchmark module is differentiator |
| Visualization | Yes (native viewer) | Yes (plotly integration) | Via Open3D export | **Have** - matplotlib-based |
| Multi-camera support | Basic | Research-focused | Yes (core feature) | **Have** - core feature with refractive modeling |
| Config-driven workflow | No (API only) | No (API only) | CLI flags | **Have** - YAML config is differentiator |
| Type hints | Partial (stubs) | Yes (throughout) | Yes | **Partial** - need to complete |
| CI/CD | Yes (GitHub Actions) | Yes (Meta infra) | Yes (GitHub Actions) | **Needed** - tests exist but no CI |
| Versioning | Semantic versioning | Semantic versioning | Semantic versioning | **Need to formalize** - in alpha |

**Key insights:**
- Documentation is table stakes - all major libs have Sphinx + examples
- Jupyter tutorials critical for research libraries (PyTorch3D model)
- Config-driven workflows are a differentiator (most libs are API-only)
- Benchmarking tools are rare - opportunity to stand out
- Refractive modeling is unique - leverage as primary differentiator

## Sources

### Production Python Package Standards
- [Scientific Python Development Guide](https://learn.scientific-python.org/development/) - Comprehensive guide on packaging, testing, docs, CI
- [Python Packaging User Guide](https://packaging.python.org/en/latest/discussions/versioning/) - Official versioning standards
- [pyOpenSci Python Package Guide](https://www.pyopensci.org/python-package-guide/) - Scientific package best practices
- [Real Python: Publishing to PyPI](https://realpython.com/pypi-publish-python-package/) - PyPI publishing guide
- [Python Poetry Guide 2026](https://devtoolbox.dedyn.io/blog/python-poetry-complete-guide) - Modern packaging tool

### Documentation Standards
- [Sphinx Documentation](https://www.sphinx-doc.org/en/master/usage/quickstart.html) - Official Sphinx guide
- [Writing Documentation - Scientific Python](https://scientific-python-cookie.readthedocs.io/en/latest/guides/docs/) - Documentation patterns
- [ReadTheDocs User Guide](https://docs.readthedocs.io/en/stable/intro/sphinx.html) - ReadTheDocs integration
- [Python Package Documentation Guide](https://inventivehq.com/blog/python-package-documentation-guide) - Comprehensive doc best practices

### Testing and Quality
- [pytest Documentation](https://docs.pytest.org/en/stable/) - Official pytest guide
- [Pytest Fixtures Complete Guide 2026](https://devtoolbox.dedyn.io/blog/pytest-fixtures-complete-guide) - Advanced pytest patterns
- [PyTorch Benchmark Documentation](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) - PyTorch benchmarking
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) - GPU optimization

### CLI Design
- [Click vs argparse Comparison](https://www.pythonsnacks.com/p/click-vs-argparse-python) - CLI framework comparison
- [Typer CLI Guide](https://towardsdatascience.com/typer-probably-the-simplest-to-use-python-command-line-interface-library-17abf1a5fd3e/) - Modern CLI patterns
- [Python CLI Tools Guide](https://inventivehq.com/blog/python-cli-tools-guide) - Building distributable CLI tools

### Logging and Progress
- [tqdm PyPI](https://pypi.org/project/tqdm/) - Progress bar library (latest 2026-02-03)
- [tqdm-loggable](https://github.com/tradingstrategy-ai/tqdm-loggable) - Logging-friendly progress bars
- [Managing Data Science Workflows](https://medium.com/@tzhaonj/managing-data-science-workflows-with-tqdm-psutil-and-logging-for-efficiency-and-transparency-fdf4b5ea9166) - tqdm + logging integration

### 3D Reconstruction Libraries
- [Open3D Documentation](https://www.open3d.org/docs/latest/introduction.html) - Production 3D library reference
- [PyTorch3D](https://pytorch3d.org/) - Research-oriented 3D library
- [COLMAP](https://colmap.github.io/) - Industry-standard SfM
- [pycolmap PyPI](https://pypi.org/project/pycolmap/0.4.0/) - Python bindings for COLMAP

### Computer Vision Ecosystem
- [Top Computer Vision Libraries 2026](https://blog.roboflow.com/computer-vision-python-packages/) - Ecosystem overview
- [Computer Vision Libraries Comparison](https://www.geeksforgeeks.org/computer-vision/computer-vision-libraries-for-python-features-applications-and-suitability/) - Feature comparison
- [PyTorch Vision Models](https://docs.pytorch.org/vision/stable/models.html) - Pre-trained model distribution

### Evaluation and Benchmarking
- [Computer Vision Benchmarks 2026](https://www.chatbench.org/computer-vision-benchmarks/) - Standard benchmarks
- [Model Performance Evaluation](https://viso.ai/computer-vision/model-performance/) - Metrics and evaluation
- [Roboflow CV Evals](https://github.com/roboflow/cvevals) - Evaluation tools

### Reproducibility
- [Ten Rules for Dockerfiles](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008316) - Container best practices
- [Singularity for Reproducibility](https://pmc.ncbi.nlm.nih.gov/articles/PMC9581025/) - HPC container workflows
- [The Turing Way - Containers](https://book.the-turing-way.org/reproducible-research/renv/renv-containers/) - Reproducible research guide

### Versioning and Release Management
- [Semantic Versioning 2.0.0](https://semver.org/) - Official SemVer spec
- [Python Versioning Best Practices](https://inventivehq.com/blog/python-package-versioning-guide) - SemVer vs CalVer
- [CHANGELOG.md Guide](https://www.pyopensci.org/python-package-guide/documentation/repository-files/changelog-file.html) - Changelog standards
- [Python Semantic Release](https://python-semantic-release.readthedocs.io/) - Automated versioning

### Data Formats and Pipelines
- [PyTorch ImageFolder](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) - Dataset organization
- [PyTorch Custom Datasets Tutorial](https://www.learnpytorch.io/04_pytorch_custom_datasets/) - Data loading patterns
- [3D File Formats Guide](https://pmrajavel.medium.com/mastering-3d-computer-vision-point-cloud-processing-mod-13-commonly-used-3d-file-formats-with-3d68d8a68cc5) - PLY, PCD, OBJ formats

### Configuration Management
- [Python TOML Guide](https://realpython.com/python-toml/) - TOML configuration
- [Configuration Files in Python](https://deepdocs.dev/configuration-files-python/) - YAML vs TOML vs JSON
- [Config Management Best Practices](https://hackersandslackers.com/simplify-your-python-projects-configuration/) - Comprehensive config guide

---
*Feature research for: AquaMVS production-ready milestone*
*Researched: 2026-02-14*
