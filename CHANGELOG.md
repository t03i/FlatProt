# CHANGELOG


## v2.1.0 (2025-04-30)

### Chores

- Add runtime analysis
  ([`1847ea2`](https://github.com/t03i/FlatProt/commit/1847ea2bd30c2ca5411113add08cc01f29207417))

### Features

- Add runtime comparison script
  ([`36775ed`](https://github.com/t03i/FlatProt/commit/36775ed4ca1aa08ee1c706b13233fd6dfd4291fa))

### Refactoring

- Ensure database path exists
  ([`283ecb0`](https://github.com/t03i/FlatProt/commit/283ecb05f87733ac5f9fc1384fcbd383d1f767a1))

- Fix parameter
  ([`35c2621`](https://github.com/t03i/FlatProt/commit/35c2621c973cb3162505966d64bf80abeee0b458))

- Remove threads
  ([`c42db6d`](https://github.com/t03i/FlatProt/commit/c42db6d823e62e1bdb3af4d4bf0774e48b7e7562))


## v2.0.6 (2025-04-29)

### Bug Fixes

- Make database more robust for jupyter notebook
  ([`804a8ff`](https://github.com/t03i/FlatProt/commit/804a8ffcd0d890458fff93639d36db96b28dddbf))

### Chores

- Add UniProt projection example tailed for Colab experience
  ([#12](https://github.com/t03i/FlatProt/pull/12),
  [`e287d01`](https://github.com/t03i/FlatProt/commit/e287d0167f13a1855868d1489c5e5dc717ea5e14))

## Summary This PR significantly improves the aesthetics and user experience of the
  uniprot_projection.py example when viewed as a Google Colab notebook. The changes focus on visual
  enhancements and interactive elements while maintaining the same functionality. ## Changes - Added
  styled section headers with custom backgrounds and borders - Implemented interactive progress bars
  for processing steps - Added color-coded status messages with appropriate icons (✅, ⚠️, ℹ️, ❌) -
  Improved error and warning displays with formatted message boxes - Enhanced SVG display with
  additional protein information card - Added visual feedback during long-running operations -
  Improved configuration display with formatted tables - Added completion banner at the end of
  execution - Added professional footer with attribution - Fixed cell structure for proper Colab
  execution

## Motivation The notebook acts as a demonstration tool for FlatProt functionality. These visual
  enhancements make it more professional-looking and easier to follow for users exploring the tool
  through Colab. The interactive elements also provide better feedback during execution.

## Testing Tested in Google Colab environment to ensure all visual enhancements render correctly and
  all functionality remains intact.

- Fix foldseek command test
  ([`b88a168`](https://github.com/t03i/FlatProt/commit/b88a168ce29e4604a7e4dc2d0af89de4a1d84c9b))

### Documentation

- Add better workflow description
  ([`8dc7a0a`](https://github.com/t03i/FlatProt/commit/8dc7a0a54cfd7194075185729f092a358ef02cfc))

- Add PR rule to cursorrules
  ([`0457605`](https://github.com/t03i/FlatProt/commit/04576056e395d4454d5f261c0f50b1d9e1517441))

### Refactoring

- Add forced yes and update UniprotID
  ([`b5ae58d`](https://github.com/t03i/FlatProt/commit/b5ae58d12ba07d1fc420176161ade8ae6c6b7bef))

- Add library install as well
  ([`c246b8b`](https://github.com/t03i/FlatProt/commit/c246b8b4fdbc8702618154ba07f87b3d1eed9d15))

- Change to atomic command
  ([`5731223`](https://github.com/t03i/FlatProt/commit/5731223e836c3ac7a30aeba02d5ef855f74c5002))

- Fix syntax issue
  ([`d765ea3`](https://github.com/t03i/FlatProt/commit/d765ea3aaa631de7c3f55b6a58bab10b44cfdb2e))

- Improve notebook style
  ([`395975c`](https://github.com/t03i/FlatProt/commit/395975cefb1c336e6878ac40d8a585553d20a27b))

- Make download jupyter compatible
  ([`be416af`](https://github.com/t03i/FlatProt/commit/be416afc7ed9d0761ef2530bb9b3cc37a047059a))

- Pipe yes
  ([`db5a8ce`](https://github.com/t03i/FlatProt/commit/db5a8ce0c4ae0bc7bc5a055bf926df614629fc13))

- Remove -qq issue
  ([`1385248`](https://github.com/t03i/FlatProt/commit/138524882feaea90645bdb2b181bacb8d1e6db38))

- Remove potentially problematic quiet
  ([`e730e9c`](https://github.com/t03i/FlatProt/commit/e730e9c116366f1937f20e90e810dab1aacf78c8))

- Revert to debian non-interactive
  ([`73c0057`](https://github.com/t03i/FlatProt/commit/73c0057d07f63a24c59f5ec9c00915225ece4242))

- Shell dssp
  ([`7019339`](https://github.com/t03i/FlatProt/commit/701933900ccd8de4672d2f631e4cec29224cb293))

- Switch to uv install
  ([`3a3f6cf`](https://github.com/t03i/FlatProt/commit/3a3f6cfed362585f11b18487b3ec2e69002f0979))


## v2.0.5 (2025-04-28)

### Bug Fixes

- Refactor DB utils and update examples
  ([`84abd4d`](https://github.com/t03i/FlatProt/commit/84abd4da0b42594ebf4402d255b97bd168c3ef3e))

This PR introduces several improvements to the handling of the alignment database and updates the
  chainsaw.py example accordingly. Key Changes: Database Utilities (src/flatprot/utils/database.py):
  Refactored DEFAULT_DB_DIR to use platformdirs for a standard, cross-platform data location
  (rostlab/flatprot). Updated DEFAULT_DB_URL to point to the correct Zenodo archive
  (alignment_db.zip). Added verify=False to the httpx client during download to mitigate potential
  SSL certificate verification errors (includes a warning log). Improved the archive extraction
  logic in download_database to: Extract to a temporary location. Specifically look for the
  alignment_db folder within the archive. Move the contents of alignment_db to the final target
  directory. Ignore other files/folders in the archive (like __MACOSX). Adjusted logging levels for
  better clarity during download/extraction. Chainsaw Example (examples/chainsaw.py): Updated the
  script to use the new flatprot.utils.database.ensure_database_available function to locate or
  download the alignment database, removing hardcoded paths. Commented out Colab-specific shell
  commands (!pip, !wget, etc.) to resolve linter errors and ensure the file is valid Python syntax.
  Added the missing import for ensure_database_available. Motivation: To make the alignment database
  download and storage more robust and platform-independent. To resolve issues related to SSL
  verification and archive structure during database download. To ensure the example script utilizes
  the centralized database handling logic. To fix linter errors in the example script.

### Chores

- Remove CNAME
  ([`8ac9db4`](https://github.com/t03i/FlatProt/commit/8ac9db4f21f74edbbf6d557c543918ace21fc8a8))

- Update connector color
  ([`979fe58`](https://github.com/t03i/FlatProt/commit/979fe584b9e831cbdfa610a1abd33cbe34b963f0))


## v2.0.4 (2025-04-28)

### Bug Fixes

- Documentation updates
  ([`9cd4af7`](https://github.com/t03i/FlatProt/commit/9cd4af74fa78306780eac5f3524a83a7c2b2e8ec))


## v2.0.3 (2025-04-28)

### Continuous Integration

- Simplify release config
  ([`0d929c3`](https://github.com/t03i/FlatProt/commit/0d929c3b15e6389b7e46b4c183158498f85d8bea))

- Update mkdocs command
  ([`6cd2768`](https://github.com/t03i/FlatProt/commit/6cd2768e5fc8c903702a751322db2ca80487d9ca))


## v2.0.2 (2025-04-28)

### Chores

- Generate example notebooks ([#9](https://github.com/t03i/FlatProt/pull/9),
  [`31ec311`](https://github.com/t03i/FlatProt/commit/31ec3113ccb0680cab91368c0c0620f269588e61))

Automated generation of Jupyter notebooks from example Python scripts. Changes detected in
  `examples/*.py`.

Co-authored-by: t03i <t03i@users.noreply.github.com>

### Continuous Integration

- Add deps to docs-release
  ([`c4bfb08`](https://github.com/t03i/FlatProt/commit/c4bfb08a9c8ba30f9920d3f0dae0a1681a335543))

- Add labels permission
  ([`4fb5d23`](https://github.com/t03i/FlatProt/commit/4fb5d237249384efc39a0f0a6483e959f1d37758))

- Add repo checkout to finalize-release
  ([`41885a6`](https://github.com/t03i/FlatProt/commit/41885a6f9235d5a0e48c1434b85aeeca7be74f2f))

- Update notebook pull request
  ([`d23e6b6`](https://github.com/t03i/FlatProt/commit/d23e6b6a645afce85f08e158607674726ccb3c4b))

- Update pull request permissions
  ([`c4ad0dc`](https://github.com/t03i/FlatProt/commit/c4ad0dcc8dea136092b1c8bd7b3ee6604b697f29))

### Documentation

- Update README
  ([`ee64954`](https://github.com/t03i/FlatProt/commit/ee649540bd2244cf777c783764d26a8ff6a8a49d))


## v2.0.1 (2025-04-22)


## v2.0.0 (2025-04-22)

### Bug Fixes

- Add - to allow for negative coords
  ([`f4df4a0`](https://github.com/t03i/FlatProt/commit/f4df4a0398c52840727cf82477daa8e5de9544cb))

- Add connection styles
  ([`771c228`](https://github.com/t03i/FlatProt/commit/771c22834f5f163e0dd05e9ac6abf6426160970f))

- Add opacity drawing
  ([`2d54351`](https://github.com/t03i/FlatProt/commit/2d5435118c28d91a5fdd1bbe79c71acc02570751))

- Better checkpoint usage
  ([`a1f1f0c`](https://github.com/t03i/FlatProt/commit/a1f1f0c0a880cbeeed8130b255fc0a0cafadde25))

- Connection calculation
  ([`5a6b9ab`](https://github.com/t03i/FlatProt/commit/5a6b9ab40bb9b4376a29aaf9d7577fa95ae6fc4a))

- Dix dssp parsing
  ([`ed3fc37`](https://github.com/t03i/FlatProt/commit/ed3fc370aea317008c60f9d12fa388cc3935da75))

- Filter for target db now working
  ([`27b12c6`](https://github.com/t03i/FlatProt/commit/27b12c61269f9305c6d3d038b65970d795a45513))

- Import issues
  ([`563ba83`](https://github.com/t03i/FlatProt/commit/563ba836ee40aa3b5cead0feb01258a834120047))

- Improve error handling in alignment and transformation matrix
  ([`466356e`](https://github.com/t03i/FlatProt/commit/466356e5cb956b4c0332fa25ff219af7f1b040bb))

- Enhanced error handling in the `align_structure_rotation` function by reintroducing specific
  handling for `FlatProtError`, ensuring consistent logging of error messages. - Added validation
  checks in the `from_array` and `from_string` methods of the `TransformationMatrix` class to ensure
  input arrays conform to expected dimensions, raising appropriate `ValueError` exceptions when
  necessary. - Updated docstrings for clarity and compliance with PEP257.

- Improve error handling in SVG rendering process
  ([`f1b310d`](https://github.com/t03i/FlatProt/commit/f1b310d23de694a59b836eb4ebe39ebeed1ea74a))

- Wrapped the SVG rendering and saving logic in a try...except block to capture and log errors
  during the rendering process. - Added logging for exceptions raised during SVG rendering,
  providing clearer context for failures. - Removed the broad try...except block in the annotation
  processing to allow errors to propagate, improving error visibility. - Updated docstrings for
  clarity and compliance with PEP257.

- Metadata
  ([`bd38488`](https://github.com/t03i/FlatProt/commit/bd3848876c76e6483abfa6c8ddb84ed3a61d0c50))

- Put params in correct position
  ([`d882b82`](https://github.com/t03i/FlatProt/commit/d882b822c207ffd101482b274f1eb0a1067828f0))

- Resolve alignment loading issues
  ([`a6b62c2`](https://github.com/t03i/FlatProt/commit/a6b62c2788e8d19a159ad18d7c263a101fe16a26))

- Resolve default style not applied
  ([`9b0e275`](https://github.com/t03i/FlatProt/commit/9b0e275a654f3544b15c200f18bbad8d149f7440))

- Resolve issues with invalid report
  ([`ed01efc`](https://github.com/t03i/FlatProt/commit/ed01efccfaf9cbbf28bfd15c316d2f4b98fb27ac))

- Resolve representative issues
  ([`f0195b5`](https://github.com/t03i/FlatProt/commit/f0195b5d62f4ff6c652696fdf62d4acc7583268c))

- Scop parser excludes discontinous domains
  ([`486937d`](https://github.com/t03i/FlatProt/commit/486937d3c8c02981f0cc899276ef16cfd6938682))

- Scop parsing for negative numbers
  ([`9fcaa2a`](https://github.com/t03i/FlatProt/commit/9fcaa2aa9cc02c04c0c813d1a103e40768341173))

- Short helix coordinate-computation
  ([`5eca824`](https://github.com/t03i/FlatProt/commit/5eca824e61dbb39c324f447cced795f906aa0e2c))

- Snakefile again
  ([`4fc8967`](https://github.com/t03i/FlatProt/commit/4fc8967f015c491f64caedfdce85dd73bcc81499))

- Snakemake
  ([`02ba248`](https://github.com/t03i/FlatProt/commit/02ba24888e34e79b6cb6bf313179608118c20371))

- Syntax error
  ([`110a916`](https://github.com/t03i/FlatProt/commit/110a916d4c6e2f1094675a0d65247dc36b549a8b))

- Update connection tests
  ([`7ea28d0`](https://github.com/t03i/FlatProt/commit/7ea28d0a67f4f195688761698fc823fcaf1aea96))

- Update db creation
  ([`1912742`](https://github.com/t03i/FlatProt/commit/19127425b5ae2d9bfb8709e6a4cc0cad50fd8da0))

- Update dependencies
  ([`fbce3a7`](https://github.com/t03i/FlatProt/commit/fbce3a756a774e062bcb24dfab501efd5bbc827e))

- Update pipeline for proper foldseek path
  ([`2e29b7f`](https://github.com/t03i/FlatProt/commit/2e29b7f3a78cd6d3dcccd1579353b731e81f0683))

- Update snakemake config
  ([`6ab9152`](https://github.com/t03i/FlatProt/commit/6ab9152c510eff9ec2987c0d33f2b22b6e9b5fd4))

- Update tests
  ([`1f93a71`](https://github.com/t03i/FlatProt/commit/1f93a7152b5a06e1043b77964a21a9b9599deaf4))

- Update tests to better connection rendering
  ([`d2ef4d7`](https://github.com/t03i/FlatProt/commit/d2ef4d7576b8d5f3957c8aa11dbff2021c455317))

- Update the foldseek rotation to be correct
  ([`47810c7`](https://github.com/t03i/FlatProt/commit/47810c7855b1c5d8382d685bb1f0d7843f631192))

### Chores

- Add command scaffold for align
  ([`27f4b92`](https://github.com/t03i/FlatProt/commit/27f4b9256c2f5935bd019eb68d5505b46767562f))

- Add fixtures and tests for multi-chain SVG rendering
  ([`ece32bf`](https://github.com/t03i/FlatProt/commit/ece32bfeaa2010814138597f17081db4eddb8e67))

- Introduced new pytest fixtures to create mock structures with two chains for testing. - Added
  tests to ensure that connections are not drawn between elements of different chains. - Enhanced
  existing tests for rendering elements to improve clarity and consistency. - Updated docstrings for
  new fixtures and tests to comply with PEP257.

- Add linter ignore
  ([`bd6315b`](https://github.com/t03i/FlatProt/commit/bd6315b171fd22a28386c7244dd152a925db6353))

- Add project integration tests
  ([`5a205cc`](https://github.com/t03i/FlatProt/commit/5a205cce1198249da873c259c9488ddcea74cd93))

- Add scaffold db creation script
  ([`25720d4`](https://github.com/t03i/FlatProt/commit/25720d47707744d30f18da510ca7d7090ecaa14b))

- Add tests for utils
  ([`e0c6e4c`](https://github.com/t03i/FlatProt/commit/e0c6e4cf4ff7abffa248a573683edeb80cfd3df7))

- Add workflow for notebooks
  ([`523ad8f`](https://github.com/t03i/FlatProt/commit/523ad8f918d588d7245c714572dc21c0625cb8b1))

- Fix svg_render test
  ([`8550c3d`](https://github.com/t03i/FlatProt/commit/8550c3d31514ad957601d93ef98535141dd818e3))

- Fix test attribute
  ([`c4ef6d5`](https://github.com/t03i/FlatProt/commit/c4ef6d5dafb9026ce3fbcfa5487bdf11f2499a48))

- Reenable large files
  ([`33f0a6d`](https://github.com/t03i/FlatProt/commit/33f0a6d06836380f5896f9bcae6445ace2d10222))

- Update
  ([`818717e`](https://github.com/t03i/FlatProt/commit/818717e3679942dc290b3d6b157c234e9bd1e1fc))

### Code Style

- Fix cmd formatting
  ([`b3ad38c`](https://github.com/t03i/FlatProt/commit/b3ad38cceaa850312396315ab1d9c85f490681b0))

- Fix example formatting
  ([`3bd37cd`](https://github.com/t03i/FlatProt/commit/3bd37cdc42a4a183dc44091851a9a91a16621c65))

- Improve cli output
  ([`633fbef`](https://github.com/t03i/FlatProt/commit/633fbef069e56a2df23ae6a1fca3a2d6cd85f783))

- Update cli description
  ([`af3faaa`](https://github.com/t03i/FlatProt/commit/af3faaa09dd4be68189ddd963b198e76bde25844))

- Update examples
  ([`04a4f91`](https://github.com/t03i/FlatProt/commit/04a4f913a4f48298ee454f902efcc495ce4917c4))

### Continuous Integration

- Fix invalid reference name
  ([`1a4a706`](https://github.com/t03i/FlatProt/commit/1a4a706fd4b7bb6028d4dbff69a2da8cc4747d0a))

- Update permissions
  ([`d8d2936`](https://github.com/t03i/FlatProt/commit/d8d2936246cbdb20ccb86514e2953cedad900c47))

### Documentation

- Add API documentation
  ([`60d4b24`](https://github.com/t03i/FlatProt/commit/60d4b2441224acaa3713a39815137db41bab4f09))

- Add example data
  ([`7193627`](https://github.com/t03i/FlatProt/commit/7193627d4d2be3950914ccd3c1e82b0d959f7a54))

- Correct foldseek command
  ([`1a8f833`](https://github.com/t03i/FlatProt/commit/1a8f83307be83ea79f5e0e6f9940e1709052242e))

- Improve description
  ([`d706a55`](https://github.com/t03i/FlatProt/commit/d706a558aa2b7e92d2a57760f392d2019d62ef8e))

- Update docs to reflect commandline interface
  ([`b6fee15`](https://github.com/t03i/FlatProt/commit/b6fee15025c6384bcbf45626536d63fa2bf660f3))

- Update documentation
  ([`04e6cca`](https://github.com/t03i/FlatProt/commit/04e6ccadfc872a763d79d2686bd5f4448f5e49e1))

- Update documentation for align
  ([`c01345b`](https://github.com/t03i/FlatProt/commit/c01345b4b53dda203bacd6d632c0444d5a2c9f85))

- Update examples
  ([`6a1fe16`](https://github.com/t03i/FlatProt/commit/6a1fe16f9c0fc6a500dbcc4faf1ad0fd76d8fb48))

- Update issue templates
  ([`dfa2d55`](https://github.com/t03i/FlatProt/commit/dfa2d55f4d25c1792c4e6ac82ed3c6c862fa6236))

- Update Readme and add CoC & Contributing
  ([`6dd9fa3`](https://github.com/t03i/FlatProt/commit/6dd9fa3000bcbd052d0379117b6012d027395209))

### Features

- Add adjacency checks for ResidueRange and ResidueRangeSet
  ([`af0259f`](https://github.com/t03i/FlatProt/commit/af0259f9eee3a8fb4931a2b8e8dfc7f1979357b3))

- Implemented `is_adjacent_to` method in `ResidueRange` and `ResidueRangeSet` classes to determine
  adjacency between ranges and coordinates. - Enhanced error handling with TypeError for invalid
  comparisons. - Added comprehensive tests for adjacency functionality in
  `tests/core/test_coordinates.py`. - Updated docstrings for clarity and compliance with PEP257.

- Add basic cmd example 3Ftx
  ([`e13ec95`](https://github.com/t03i/FlatProt/commit/e13ec959e40d5855365cb5b55fa93b86d5347862))

- Add db download functionality
  ([`6ad7e6e`](https://github.com/t03i/FlatProt/commit/6ad7e6eb6d0740ab06c5dee63a73c29c3271b3bb))

- Add domain reporting
  ([`f8d230f`](https://github.com/t03i/FlatProt/commit/f8d230f62972f4dc24b281d013ad656ab5500d1d))

- Add domain transformation utilities and update examples
  ([`b1c1043`](https://github.com/t03i/FlatProt/commit/b1c1043e6af91051499a0b82ca4c47d935a53927))

- Introduced `DomainTransformation` class to encapsulate transformation matrices for specific
  protein domains. - Implemented `apply_domain_transformations_masked` function to apply
  transformations using boolean masks, ensuring overlapping domains are handled correctly. - Added
  `create_domain_aware_scene` function to generate scenes with elements grouped by domain, improving
  spatial arrangement and visualization. - Updated examples to reflect new domain transformation
  functionalities and ensure clarity in usage. - Enhanced error handling and logging throughout the
  new utilities. - Updated docstrings and added typing annotations to comply with PEP257 standards.

- Add example for overlay and annotations
  ([`de2e0ca`](https://github.com/t03i/FlatProt/commit/de2e0ca6c49293a5d5a7458868ab9d38e0fa2753))

- Add improved chainsaw example
  ([`3e3cbb1`](https://github.com/t03i/FlatProt/commit/3e3cbb15d8882fa830183db8d35d05216a3ec0d6))

- Add initial db pipeline
  ([`d7bb462`](https://github.com/t03i/FlatProt/commit/d7bb46216d3e477fc42556b3090da47e52a8a5b4))

- Add runtime analysis
  ([`4e9ec82`](https://github.com/t03i/FlatProt/commit/4e9ec8245d149c36af5ca746e4f753641af1b709))

- Add uniform projection
  ([`4e3f185`](https://github.com/t03i/FlatProt/commit/4e3f1854b9f3ee13eb742e69240ecd4dc54d3981))

- Enhance SVG rendering with connection point calculations
  ([`fa29171`](https://github.com/t03i/FlatProt/commit/fa2917176839ca06d0542f8989e847e255fbb8fa))

- Added `_prepare_render_data` method to `SVGRenderer` for pre-calculating 2D coordinates and
  connection points for structure elements. - Introduced new utility functions in `svg_structure.py`
  to calculate connection points for coils, helices, and sheets. - Updated drawing functions to
  utilize the new connection point calculations. - Improved error handling and logging for data
  preparation failures. - Enhanced tests in `test_svg_renderer.py` to cover new rendering logic and
  connection point assertions. - Updated docstrings for clarity and compliance with PEP257.

- Integrate staging branch updates introducing breaking changes
  ([#8](https://github.com/t03i/FlatProt/pull/8),
  [`02d2488`](https://github.com/t03i/FlatProt/commit/02d2488806fe072a48f29c60f29433b426ccb048))

This pull request merges the accumulated features, refactors, and bug fixes from the staging branch
  into main. It represents a significant update to the application's core functionality and
  structure. Summary of Changes: Integration of various features and improvements developed and
  tested on the staging branch. Refactoring of key components to enhance modularity,
  maintainability, and performance. Updates to dependencies and build processes. Improvements to
  testing infrastructure and documentation.

BREAKING CHANGES: This merge introduces significant breaking changes due to [briefly mention the
  reason for the breaking changes, e.g., major API redesign, data model restructuring, CLI command
  signature changes]. Compatibility with previous versions deployed from main is broken. Users and
  downstream systems will need to adapt to the new interfaces/structures.

### Refactoring

- Add alignment step to commands
  ([`a0abe56`](https://github.com/t03i/FlatProt/commit/a0abe56f39f9c25e2528b5e940808c9a46fa9a37))

- Add basic scene tests
  ([`4b6e423`](https://github.com/t03i/FlatProt/commit/4b6e42347dbddeee0028d3de23d98a5ef2848a77))

- Add connections class
  ([`c33d01f`](https://github.com/t03i/FlatProt/commit/c33d01f951c1426a14d6e196e97be8b6140e07c5))

- Add debug output
  ([`bef3e4b`](https://github.com/t03i/FlatProt/commit/bef3e4bfb31aa2401c084ffde89ec3d429812f03))

- Add error handling to scop parsing
  ([`9dbb2d7`](https://github.com/t03i/FlatProt/commit/9dbb2d736e99bc256c6349bf637ab367868bce1d))

- Add family-identity alignment mode as default
  ([`ebab00e`](https://github.com/t03i/FlatProt/commit/ebab00e111b77c4aebbb589e8a1288755a718c33))

- Add id calculation for annotations
  ([`030904e`](https://github.com/t03i/FlatProt/commit/030904e21667d58d960a8c974f5b13577e80e999))

- Add label text to svg
  ([`fe8cf11`](https://github.com/t03i/FlatProt/commit/fe8cf11cd136815ed4b4c3c61eb76dda00a0a374))

- Add logfiles
  ([`982adf6`](https://github.com/t03i/FlatProt/commit/982adf632571f008f693a86fc76623c418993d23))

- Add logging output config
  ([`9aad01a`](https://github.com/t03i/FlatProt/commit/9aad01a20bda544911c7768f60c7eba3eede26aa))

- Add metadata to db
  ([`3aedb43`](https://github.com/t03i/FlatProt/commit/3aedb43a0681641fe39972eae9a21b8a4d984744))

- Add proper group transforms
  ([`4639237`](https://github.com/t03i/FlatProt/commit/46392373952c5503e4a9adcb4c1affe828ba51bd))

- Add render tests
  ([`3632b8c`](https://github.com/t03i/FlatProt/commit/3632b8cad6094a033b85431e679cb7b242509f17))

- Add scene integration tests
  ([`36d7341`](https://github.com/t03i/FlatProt/commit/36d73410088fed454c6705d9036c3fc3431b8874))

- Add scene_utils for scene creation
  ([`162c61f`](https://github.com/t03i/FlatProt/commit/162c61f3bbf752ade8e73662b50143c4604ced4b))

- Add structure id and proper dict iteration
  ([`aa1479a`](https://github.com/t03i/FlatProt/commit/aa1479ab1f3ea58d5be3b01526b8ef33e0599e9a))

- Add tests for core elements
  ([`d3813a3`](https://github.com/t03i/FlatProt/commit/d3813a3e44e3ea45de2ca3b34bab637585a6978d))

- Add vectorized coordinate manipulation to structure
  ([`04664d1`](https://github.com/t03i/FlatProt/commit/04664d17df29c27d272febaded2bc78da341ce86))

- Add version
  ([`0864731`](https://github.com/t03i/FlatProt/commit/0864731bf12d460fdadab8d6b879ab7a461b3d73))

- Adopt stateless architecture and enhance documentation
  ([#7](https://github.com/t03i/FlatProt/pull/7),
  [`b1afec6`](https://github.com/t03i/FlatProt/commit/b1afec6d9a3e0997d3414450c528b85dc8f0e359))

# Refactor to Stateless Architecture and Enhance Documentation

This pull request introduces a significant refactoring of the codebase to adopt a more stateless
  architecture. This change aims to improve predictability, testability, and scalability.

Key changes include:

* **Stateless Refactoring:** Core components have been redesigned to operate without maintaining
  internal state where possible. This simplifies the flow of data and reduces potential side
  effects. * **Extensive Documentation:** Added comprehensive docstrings to functions and classes
  across the project, adhering to PEP 257 conventions. Updated existing documentation to reflect the
  architectural changes. * **New Examples:** Included several new example scripts in the `examples/`
  directory to demonstrate the usage of the refactored components and showcase common workflows.
  These examples are designed to be easily convertible to Jupyter notebooks using Jupytext.

**Review Focus:**

* Please review the architectural changes for adherence to stateless principles. * Verify the
  clarity and accuracy of the new documentation and examples. * Ensure the refactored code maintains
  existing functionality and passes all tests.

This refactor sets a foundation for future development by improving the overall structure and
  maintainability of the codebase.

- Allow both cif and pdb
  ([`e6d6b8d`](https://github.com/t03i/FlatProt/commit/e6d6b8d3ca82643ba7a425ef2848c58d7b308aa2))

- Better 3ftx display
  ([`8e96271`](https://github.com/t03i/FlatProt/commit/8e962711f211f65138d44e7f1d5a098851da8eeb))

- Bump test family number
  ([`a827fdc`](https://github.com/t03i/FlatProt/commit/a827fdcf1109eda4c0f7e6d3f0fdac1173e8ab3b))

- Change db to be ignored per-default
  ([`f377fee`](https://github.com/t03i/FlatProt/commit/f377feefd0ea1a1a6bb27f2145b7c708fc2e3433))

- Change index off by one issues
  ([`e64ad11`](https://github.com/t03i/FlatProt/commit/e64ad111417d09ad5145d121225a7618f09ff0e3))

- Change out to tmp
  ([`118487e`](https://github.com/t03i/FlatProt/commit/118487e8d48c68fe7519dd05ebb055355e3f5b91))

- Change to secondary_structure_type
  ([`d5bc870`](https://github.com/t03i/FlatProt/commit/d5bc870040e36937d20a73d091564eafff230c57))

- Clean up pipeline def
  ([`46db50f`](https://github.com/t03i/FlatProt/commit/46db50f64c58fe0382c7521bff7af5a0f1d5bb58))

- Cleanup transformation structure
  ([`3440429`](https://github.com/t03i/FlatProt/commit/3440429c4b1fb2ad3fc33f342940ff365fa38e31))

- Enhance error handling and update annotations
  ([`597ae7f`](https://github.com/t03i/FlatProt/commit/597ae7fa6eeaa92db31c9eee5b4f9f38107755d8))

- Added `CoordinateCalculationError` to the core error handling for better clarity in error
  propagation. - Updated the `__init__.py` file to include the new error class in the module
  exports. - Refactored the annotation tests to improve clarity and ensure proper error handling
  during coordinate calculations. - Enhanced existing tests to cover scenarios where annotations
  return None or invalid shapes, ensuring robust error reporting. - Updated docstrings across
  various files to comply with PEP257 standards.

- Enhance transformation matrix methods and update docstrings
  ([`4a2be7f`](https://github.com/t03i/FlatProt/commit/4a2be7fdb13e87322bb845bac58bd05e18ecc5cc))

- Refactored the `get_aligned_rotation_database` function to improve clarity in rotation matrix
  application. - Updated the `calculate_inertia_transformation_matrix` function's docstring to
  better describe the transformation process and return values. - Renamed the `combined_rotation`
  method to `before` in the `TransformationMatrix` class for improved clarity, and added a new
  `after` method for combining transformations. - Enhanced the `apply` method in the
  `TransformationMatrix` class with additional input validation and updated docstring for clarity. -
  Updated tests for inertia transformation to reflect changes in the transformation logic and ensure
  proper validation of expected outcomes. - Ensured all docstrings comply with PEP257 standards and
  added typing annotations where necessary.

- Finalize core structure API
  ([`aa8c4b8`](https://github.com/t03i/FlatProt/commit/aa8c4b8ce4d02583848832a77d2c3e1e74a57268))

- Fix example
  ([`2cd8aa6`](https://github.com/t03i/FlatProt/commit/2cd8aa66a058fde09c4b2d416cd99421801160c5))

- Fix issue
  ([`35a3826`](https://github.com/t03i/FlatProt/commit/35a3826b18967320e769a1c94df26143dfb09cfd))

- Fix issues
  ([`b6e61d9`](https://github.com/t03i/FlatProt/commit/b6e61d9e77eb9ef0ab264e94de8b5e32dd6d3e19))

- Fix issues
  ([`4a4cd28`](https://github.com/t03i/FlatProt/commit/4a4cd28c73c31cc9165ec77d095e45072c46be13))

- Fix missing input rule issues
  ([`e14649d`](https://github.com/t03i/FlatProt/commit/e14649df8dac3bf73a9ed3ede150773f16e63c1c))

- Fix pipeline end to end
  ([`d2cf57c`](https://github.com/t03i/FlatProt/commit/d2cf57cb0e02691229e0cbef5eb2a81b4b76cb5f))

- Fix structure tests
  ([`f9594ea`](https://github.com/t03i/FlatProt/commit/f9594ea8a073609bb70a3c928aabf427a154f057))

- Fix styles test
  ([`db49238`](https://github.com/t03i/FlatProt/commit/db492386bca39ab6c8cddbe9c189d7e6bab8c2a1))

- Foldseek threads to 4
  ([`14937ed`](https://github.com/t03i/FlatProt/commit/14937ed4753737413619111588f0d23b8f8dd05a))

- Global style rename
  ([`9dff8ba`](https://github.com/t03i/FlatProt/commit/9dff8ba103c206ea6354fec2163eb0b13989c514))

- Implement chain coordinates
  ([`84b1621`](https://github.com/t03i/FlatProt/commit/84b162142cc86a64a4bad4bc6ef0e941f5fdada9))

- Implement new structure in commands
  ([`0a9eeea`](https://github.com/t03i/FlatProt/commit/0a9eeea129c511e911c78ed5b0b6584e9818b679))

- Improve annotation calculation logic
  ([`f0db8d7`](https://github.com/t03i/FlatProt/commit/f0db8d7a024e1acf875891c63fe193ef223f9008))

- Improve clarity for projection
  ([`13037da`](https://github.com/t03i/FlatProt/commit/13037dafaf2720cc6fb6ccfa75c94103313d103d))

- Improve commands naming
  ([`b03fcf2`](https://github.com/t03i/FlatProt/commit/b03fcf2d8671492c50be2e7d322915b098dbc319))

- Improve naming
  ([`0925eef`](https://github.com/t03i/FlatProt/commit/0925eef63071f6b1eb9c7eda76410d6af92c2f4f))

- Improve param clarity
  ([`bdf8a86`](https://github.com/t03i/FlatProt/commit/bdf8a8678413d67fac09460cc0f52fc575e9a37a))

- Improve scene to handle the annotation coordinates.
  ([`96400c5`](https://github.com/t03i/FlatProt/commit/96400c536f080ad51da2ae177f3e2adcdd5706d2))

Extend the scene class and refactor and extend tests to cover the functionality

- Move alignment results to avoid circular import
  ([`3bcd220`](https://github.com/t03i/FlatProt/commit/3bcd2209dadac5e16881e85290709446a347a15c))

- Move annotation tests to new structure
  ([`3eb5928`](https://github.com/t03i/FlatProt/commit/3eb5928f333b6f70284dd30850f950ea6bdcc755))

- Move checkpoint to pdb download
  ([`06f25aa`](https://github.com/t03i/FlatProt/commit/06f25aa08646120d378722461266a4dcb8691200))

- Move data wrangling to utils package
  ([`20f8313`](https://github.com/t03i/FlatProt/commit/20f831391cf4ed63a25a07a74f4274c4bf9ff8dc))

- Move style to notebook
  ([`5b2f73b`](https://github.com/t03i/FlatProt/commit/5b2f73baea504607f827ae9413c1c6cda637f6ef))

- Move to config file
  ([`83714f7`](https://github.com/t03i/FlatProt/commit/83714f7a94072df351e587535cfd8b7eb951be79))

- Overhaul rendering logic to work with improved structure and scene
  ([`cc5447f`](https://github.com/t03i/FlatProt/commit/cc5447fcbe38de6c4701a8cbefbdb88cac14b43e))

- Overhaul scene graph to be more linear and streamlined
  ([`ec3d400`](https://github.com/t03i/FlatProt/commit/ec3d4007371b443ad9e1290043ca3a721a2c6377))

- Proper tempfile handling
  ([`7790a22`](https://github.com/t03i/FlatProt/commit/7790a22cf01b69dbee94ec96cf7295549c78824d))

- Remove coordinate manager
  ([`6bd1649`](https://github.com/t03i/FlatProt/commit/6bd1649453372d5047af0dbd2f13b4cfcdd1f53e))

- Remove coordinate manager for coordinate transformation
  ([`8174a24`](https://github.com/t03i/FlatProt/commit/8174a2406b89231788f353a5889f09f8998178af))

- Remove obsolete tests
  ([`19848de`](https://github.com/t03i/FlatProt/commit/19848de157a8d7c52d14159dcb44194549682c59))

- Remove old db
  ([`71ddbcf`](https://github.com/t03i/FlatProt/commit/71ddbcfeea3c8a8d9526edde90fbd6631c7b9d1d))

- Remove style file
  ([`5c55f54`](https://github.com/t03i/FlatProt/commit/5c55f548e422ed5bc7a599ff565ee275a680708c))

- Remove target_residue_set
  ([`1db2fa6`](https://github.com/t03i/FlatProt/commit/1db2fa6ab51305cdf5125206960445806b1dd37f))

- Remove unused uniform projection
  ([`0909676`](https://github.com/t03i/FlatProt/commit/09096763d1328baaeb0c9ac6e4df28d25ce24ec8))

- Rename scene node to element
  ([`cf985bf`](https://github.com/t03i/FlatProt/commit/cf985bf6edeabc00fbdd40f3b8eb3eaec6837528))

- Rename to snakefile
  ([`03988c2`](https://github.com/t03i/FlatProt/commit/03988c2588a5743cf91f0a76095394d7cc261323))

- Status to json
  ([`bd52157`](https://github.com/t03i/FlatProt/commit/bd52157614597f18739d2fe6acb8f798b08e04f5))

- To allow for domain based
  ([`63208ff`](https://github.com/t03i/FlatProt/commit/63208ffe2ba7a85d4acaee4e8081d4fe6707ba94))

BREAKING CHANGE: overhaul coordinate management

- Tune CPU requirement for faster speeds
  ([`083cdee`](https://github.com/t03i/FlatProt/commit/083cdee7a2cb6993bdb4ed0337953a8d37ebab5e))

- Update annotation handling and error propagation
  ([`66328db`](https://github.com/t03i/FlatProt/commit/66328db176ec0202fc63e30e9c5228895f1c89c3))

- Refactored annotation classes to utilize a new `CoordinateResolver` for improved coordinate
  resolution. - Introduced `TargetResidueNotFoundError` in the scene error handling to enhance
  clarity in error reporting. - Updated the `get_coordinates` method in annotation classes to accept
  a `CoordinateResolver` instance, ensuring consistent coordinate retrieval. - Enhanced error
  handling in the `SVGRenderer` to catch specific errors during annotation processing. - Removed
  outdated tests related to scene coordinate delegation, as this logic is now managed by annotations
  and the resolver. - Updated docstrings across various files to comply with PEP257 standards.

- Update annotations
  ([`8fb8a52`](https://github.com/t03i/FlatProt/commit/8fb8a52c1fd8a52cfe30874d1c184d455b2bcf3f))

- Update annotations parser to include style
  ([`0e8db7c`](https://github.com/t03i/FlatProt/commit/0e8db7c041a1d4512cd772e688e073a4452c79b9))

- Update db id
  ([`a455d5d`](https://github.com/t03i/FlatProt/commit/a455d5dabc59869bde7771f5280b6edbdaf60609))

- Update db path
  ([`8226d27`](https://github.com/t03i/FlatProt/commit/8226d275588555af88537c0d75ecc10e0c1480c5))

- Update domain implementation
  ([`af32432`](https://github.com/t03i/FlatProt/commit/af324321c9f509a81031b32ce7f43c829f1156e6))

- Update extract domain output
  ([`0a53495`](https://github.com/t03i/FlatProt/commit/0a534950768824faf3c11df2217f85dcbda062a4))

- Update imports
  ([`04a18ad`](https://github.com/t03i/FlatProt/commit/04a18adab667640f1fce47d25204b03e7fa976f2))

- Update module paths
  ([`8714fea`](https://github.com/t03i/FlatProt/commit/8714feabebbb6b291977c6eccf6cae87f8a5a658))

- Update numpy typing for projection parameters
  ([`dddc5a6`](https://github.com/t03i/FlatProt/commit/dddc5a6ec5665d13a24bf6c4536bdefd67dfdf0e))

- Update overlay to regular commands
  ([`7330b9a`](https://github.com/t03i/FlatProt/commit/7330b9aa9cd366e8b1fed33305a78d049b90bfac))

- Update projection test and upgrade to better numpydantic library
  ([`c046245`](https://github.com/t03i/FlatProt/commit/c0462452b7cd7bc90b476d0c95e8bf852380667f))

- Update style parser to not use manager
  ([`aa4e10d`](https://github.com/t03i/FlatProt/commit/aa4e10db7323e022bb705a3f3a20660ed50a8a7d))

- Update styling
  ([`d5273e1`](https://github.com/t03i/FlatProt/commit/d5273e1ff6980541006b71e91bd334244da6f571))

- Update the snakemake pipeline
  ([`4e83358`](https://github.com/t03i/FlatProt/commit/4e83358c19f4244e655e30dbb77269813b9b61cb))

- Update transformation tests and fix transformation apply issue
  ([`1e50155`](https://github.com/t03i/FlatProt/commit/1e50155d1037f1e54a904f0de2e7d7ca417c4cf1))


## v1.0.0 (2025-03-17)

### Continuous Integration

- Fix workflow definition
  ([`a47e74c`](https://github.com/t03i/FlatProt/commit/a47e74c7d6c891ae43190fa6aca7a3f8f00a4442))

BREAKING CHANGE: overhauls CLI API, installation, and project structure

### Breaking Changes

- Overhauls CLI API, installation, and project structure


## v0.2.0 (2025-03-17)

### Bug Fixes

- Add h5py dep
  ([`a5ba429`](https://github.com/t03i/FlatProt/commit/a5ba429b97bd78bdf35225aaa5e5d92fa3bf9a73))

- Annotation parsing
  ([`0bc3635`](https://github.com/t03i/FlatProt/commit/0bc36353b78ee06107946609dc721dd66c79ef02))

- Missing StyleManager issue
  ([`bcd98a5`](https://github.com/t03i/FlatProt/commit/bcd98a52f6e98a598b8ebb2849c8de130045a251))

- Structure mocking
  ([`0f75ee4`](https://github.com/t03i/FlatProt/commit/0f75ee4fa27699ac7523613094b6a1ba5af8d310))

- Update matrix dimensions to be properly handled
  ([`0d6fc02`](https://github.com/t03i/FlatProt/commit/0d6fc027f928fcdf8b254054c5f5019dba7bc01c))

### Chores

- Update lockfile
  ([`ae53666`](https://github.com/t03i/FlatProt/commit/ae53666cbb8b6baf34bf48a1cc1f4a5bbb6c6f96))

- Update uv version
  ([`50ac19c`](https://github.com/t03i/FlatProt/commit/50ac19c42b839627393b96460df96f15374ea4d8))

### Continuous Integration

- Add debug to pytest
  ([`3442168`](https://github.com/t03i/FlatProt/commit/3442168ec6a0bafb45f6d129af60069033f6c4fe))

- Fix build commit
  ([`a70a15b`](https://github.com/t03i/FlatProt/commit/a70a15b2abd39c89266df08466aa2b558e2dca5a))

- Rename release files
  ([`1cfa377`](https://github.com/t03i/FlatProt/commit/1cfa377b2452c750f7479b69e16d75102bd3a48c))

- Update python tests on staging
  ([`b03aaac`](https://github.com/t03i/FlatProt/commit/b03aaac67b8ddc5efebf16d20ea0ad401697ecdd))

### Documentation

- Add basic CLI documentation
  ([`f3a5f6a`](https://github.com/t03i/FlatProt/commit/f3a5f6a3ca17ee426f0a7cdb475d9c0d11ebe77f))

- Add cursorrules
  ([`d71abc6`](https://github.com/t03i/FlatProt/commit/d71abc6c4c39d36aa546014f1b8da884881c9a20))

- Add example toml files
  ([`b0fccc6`](https://github.com/t03i/FlatProt/commit/b0fccc68dba23dbdedceb4b22b7a363c4ddd9bc1))

- Add structure documentation
  ([`f2b84b2`](https://github.com/t03i/FlatProt/commit/f2b84b29a13798a2b1065946a515a08b6b9279f0))

- Fix numbering
  ([`8786a8b`](https://github.com/t03i/FlatProt/commit/8786a8b021a6f2b33d1eb8e192eb9ca85cbc5896))

- Update example to work with 3FTx
  ([`0281cc8`](https://github.com/t03i/FlatProt/commit/0281cc897a836c7c2704ceebbcc3eb1f5f2edbe6))

- Update README
  ([`969da70`](https://github.com/t03i/FlatProt/commit/969da707d41f934301ba7c4898575079663c4532))

### Features

- Add annotation parsing and validation
  ([`e2dc6c7`](https://github.com/t03i/FlatProt/commit/e2dc6c7bda4a5a489abca712bbc412deba58ef31))

- Add annotation parsing from toml
  ([`844dc30`](https://github.com/t03i/FlatProt/commit/844dc30728a31f9151d5725eb32be42e351ec9f4))

- Add cli scaffolding
  ([`9564b9b`](https://github.com/t03i/FlatProt/commit/9564b9b559116a130632fee41104ca0ead96b852))

- Add matrix loader
  ([`e271745`](https://github.com/t03i/FlatProt/commit/e271745605b7a8182c1a9e79cb9cd7690278023d))

- Add style parsing
  ([`536684c`](https://github.com/t03i/FlatProt/commit/536684c3d291a7ef6256d6182e683f8bf5dd630b))

### Refactoring

- Adapt file test to refactored code
  ([`8832568`](https://github.com/t03i/FlatProt/commit/8832568028271dabbc0b0aff9c3f5d64bbdb449f))

- Add coordinate manager tests
  ([`3289be4`](https://github.com/t03i/FlatProt/commit/3289be4ae7e8e3056da525956b7f6a025a9924df))

- Add dssp option and validation
  ([`3b06594`](https://github.com/t03i/FlatProt/commit/3b0659488ba83319cd23d5677d70a8bd6a3b07a9))

- Add integration tests
  ([`ce4bfc5`](https://github.com/t03i/FlatProt/commit/ce4bfc53903fb22ce0cef3863d093245d5321c5f))

- Add logging and verbosity control
  ([`ca577b5`](https://github.com/t03i/FlatProt/commit/ca577b5824524fbdf21be9aee5705952f49a1875))

- Add scene utils tests
  ([`9db6bb1`](https://github.com/t03i/FlatProt/commit/9db6bb1a86beebdcad1348ad81d684d2f2b18f59))

- Add string representation
  ([`d4002cd`](https://github.com/t03i/FlatProt/commit/d4002cdb1b5a87955050fa8bac8f118da1108296))

- Add structure validation and remove tests
  ([`e9df138`](https://github.com/t03i/FlatProt/commit/e9df1380c4a0d222ce6f1c48f81a56916c4e5029))

- Add svg creation tests
  ([`0fe06a1`](https://github.com/t03i/FlatProt/commit/0fe06a152c4e530a7939dbb24d4682e20fd96cbe))

- Add tests for style utils
  ([`9ff21e8`](https://github.com/t03i/FlatProt/commit/9ff21e89a414c29370e2ef969d7a5252acf3e2cb))

- Add transformation utils
  ([`bbbccab`](https://github.com/t03i/FlatProt/commit/bbbccab2d267729a5e55be9eceecb2622792d3b8))

- Allow for console output of svg
  ([`1d7f925`](https://github.com/t03i/FlatProt/commit/1d7f92508e88783c3fbfe3817e450035fc3a36bc))

- Centralize cli fixtures
  ([`1f6af7a`](https://github.com/t03i/FlatProt/commit/1f6af7abf19e6a2539666dafc6f2cbd8add870ee))

- Change utils structure
  ([`137af54`](https://github.com/t03i/FlatProt/commit/137af54066941efc4aa4697466ce98bbe24b7aed))

- Fix integration tests
  ([`a9aabfc`](https://github.com/t03i/FlatProt/commit/a9aabfc2567008ba9c2a0f205016d6e49f54d8ca))

- Fix scene structure parsing
  ([`684095c`](https://github.com/t03i/FlatProt/commit/684095ccec9cb5204920b6d6036ca02a2a59972f))

- Fix styling
  ([`f99be5c`](https://github.com/t03i/FlatProt/commit/f99be5c227b6ce40c7e4ea16ffea45d9995f8b31))

- Fix test cases
  ([`44ace30`](https://github.com/t03i/FlatProt/commit/44ace304bede4545935ef251d97d2242ba3b6ad6))

- Fix test to work with point index
  ([`cf993e8`](https://github.com/t03i/FlatProt/commit/cf993e8d4ce231c3065ee5c497c278cfe745484f))

- Fix the local index calculation
  ([`b642eec`](https://github.com/t03i/FlatProt/commit/b642eecd148cdfcd5a34c590af7b9fa61667a66c))

- Fix utils test to accept logger
  ([`b54633e`](https://github.com/t03i/FlatProt/commit/b54633ea53b3a5b6c9da9e19821bf11e842e46a3))

- Implement detailed error logging
  ([`e17e45a`](https://github.com/t03i/FlatProt/commit/e17e45af6abc1d8d6b557c01c0e37a70e1786756))

- Improve tests to be behavior oriented
  ([`573a0d1`](https://github.com/t03i/FlatProt/commit/573a0d16817304fcdb22c030d9d966b31d6e2385))

- Move composer to utils
  ([`54c95e8`](https://github.com/t03i/FlatProt/commit/54c95e80d87df8f683edc83955051d03e6f9a22b))

- Move errors into io
  ([`f55265b`](https://github.com/t03i/FlatProt/commit/f55265b2cfce7b64a9dc41058be9eca7d0ae0343))

- Move tests to folder structure
  ([`cad5923`](https://github.com/t03i/FlatProt/commit/cad59235bb2a6857c372a4cf6f28fe94f62f36b9))

- Move to pytest-mock
  ([`2fb88cf`](https://github.com/t03i/FlatProt/commit/2fb88cf382c621f47da256b317c9f740f50706c4))

- Remove debug output
  ([`c668841`](https://github.com/t03i/FlatProt/commit/c668841136b733ebba854868fe9bad1a819a8cd1))

- Remove debug print
  ([`3fa2ca9`](https://github.com/t03i/FlatProt/commit/3fa2ca941ee08c8778ef477f50e365d9fa3b89ed))

- Remove unused import
  ([`d05b17b`](https://github.com/t03i/FlatProt/commit/d05b17b13c3a0a3d8113e6bf0da0b48987d86ad5))

- Rename to point index
  ([`ff84fab`](https://github.com/t03i/FlatProt/commit/ff84fab33b20bbef56b29fdca89ffdfc5de95696))

- Switch transformation loading to coordinate manager
  ([`02d3835`](https://github.com/t03i/FlatProt/commit/02d38359cd72e14bd4061f645703869092a6f0ef))

- Update function names
  ([`9c99757`](https://github.com/t03i/FlatProt/commit/9c9975747d67eb9b5c3e32a5a5b015d11dea93dd))

- Update io tests to work with changed annotation parser
  ([`0f00d67`](https://github.com/t03i/FlatProt/commit/0f00d677f0c53b19ea2918e1e09549da03c5d7b1))

- Update test for better annotation parsing
  ([`01ba041`](https://github.com/t03i/FlatProt/commit/01ba041195deb9a1b0660eae121ebe10b2de51eb))


## v0.1.0 (2025-02-25)

### Bug Fixes

- Adhere to proper rotation standards
  ([`ffbc370`](https://github.com/t03i/FlatProt/commit/ffbc3707925e85824470f6da180e51fdbee69fdd))

- Change name to match github action
  ([`0fc916a`](https://github.com/t03i/FlatProt/commit/0fc916a02d39e884ca0da05bc46fbae7a718bc18))

- Correct color type import
  ([`0ea0ba1`](https://github.com/t03i/FlatProt/commit/0ea0ba12e15751d2be6012033f708b8dd24bf525))

- Data ownership for secondary structure
  ([`6bdf6f5`](https://github.com/t03i/FlatProt/commit/6bdf6f5d02a01142ecb695033b92889967253396))

- Database tests
  ([`a964c42`](https://github.com/t03i/FlatProt/commit/a964c42d5f3e4181850da9c36e7eac48e7319d13))

- Fix helix visualization
  ([`b0e7661`](https://github.com/t03i/FlatProt/commit/b0e766178544a433c75836433e7888111eb82e83))

- Foldseek test
  ([`c842464`](https://github.com/t03i/FlatProt/commit/c84246406935421d6b4c34de9b4493e3ea612333))

- Helix and sheet working again
  ([`0e62f78`](https://github.com/t03i/FlatProt/commit/0e62f78847a1238ee55068a14418f3f577f01b5e))

- Improve rotation
  ([`e48447e`](https://github.com/t03i/FlatProt/commit/e48447eebcedf21ed87846e5241389117e001e79))

- Parameter passing
  ([`66f4232`](https://github.com/t03i/FlatProt/commit/66f4232157cafe3114854233f9bb085b683ed14c))

- Projection works correctly now
  ([`783e00e`](https://github.com/t03i/FlatProt/commit/783e00e0a9f85d249f044c7574625c5a6a91d713))

- Projector separated out
  ([`1a6c884`](https://github.com/t03i/FlatProt/commit/1a6c8840c255d7fea7048a07bb240b2f5208bbb6))

- Refactor coordinate transformation and test
  ([`09079fa`](https://github.com/t03i/FlatProt/commit/09079fab818b64f6ca3f6c3e6ab6509357c117eb))

- Render pipeline
  ([`c3a079c`](https://github.com/t03i/FlatProt/commit/c3a079c21b9a81f95abcd6aaadd3a6559cc371f6))

- Resolve element import issue
  ([`cd2868e`](https://github.com/t03i/FlatProt/commit/cd2868e23d76813e3c8e3d77ef1678a46c71e45e))

- Rotation algorithm now follows proper axis alignment
  ([`c579231`](https://github.com/t03i/FlatProt/commit/c579231870f5bb2128262940ed5ca62910e76471))

- Update projector to properly work
  ([`1779e0d`](https://github.com/t03i/FlatProt/commit/1779e0de1e42573b13a741ccca949e1dad7d5b09))

- Update tests and fix problems with implementations
  ([`5f5a36c`](https://github.com/t03i/FlatProt/commit/5f5a36cbb02324644f235e72da8c688198ff7be5))

- Utils test
  ([`bf349e5`](https://github.com/t03i/FlatProt/commit/bf349e5c54fde3649ab9512623635a2e0057953e))

### Chores

- Add 3FTx example data
  ([`23adf33`](https://github.com/t03i/FlatProt/commit/23adf3397db8b0215d3ce6a1e332f8d4a67fb257))

- Add basic tests for visualization
  ([`0c412a5`](https://github.com/t03i/FlatProt/commit/0c412a57849f1fbfa45a95026ca492e81a753a46))

- Add github actions
  ([`f1044af`](https://github.com/t03i/FlatProt/commit/f1044af19fcf973463146b6fbb32bd5b3e6b3da1))

- Add pytest
  ([`aec125b`](https://github.com/t03i/FlatProt/commit/aec125be4a6b382d0acc203b7c6fd41937c91b15))

- Add tests for projectors
  ([`9902ce4`](https://github.com/t03i/FlatProt/commit/9902ce4005fd6b179e113a18762f5dba979dc91a))

- Change import names
  ([`360586a`](https://github.com/t03i/FlatProt/commit/360586a8e3d7827b1ed2b06e91dfa9a4147adc8d))

- Change paths
  ([`c1ca4bb`](https://github.com/t03i/FlatProt/commit/c1ca4bbfeee3aec9e0d9070fbd28491ef8e20031))

- Ignore auto gen files
  ([`01fc17e`](https://github.com/t03i/FlatProt/commit/01fc17ef75abc947979127928c645a05cb1b6c00))

- Make projection tests more meaningful
  ([`9136460`](https://github.com/t03i/FlatProt/commit/9136460a528241f12c54c75562108c1577850acb))

- Move old files out of code base
  ([`103ed60`](https://github.com/t03i/FlatProt/commit/103ed600663bf6c1cfb08193e01a678ba5ea5063))

- Remove unnecessary files
  ([`61628e6`](https://github.com/t03i/FlatProt/commit/61628e6c823ed649afaaa26a502637abae1e6451))

- Rename elements to structure
  ([`1ce8a76`](https://github.com/t03i/FlatProt/commit/1ce8a767caa16fc0c8e6508c3d0b1ade3e724e33))

- Rename example
  ([`8058833`](https://github.com/t03i/FlatProt/commit/805883397e32e0028f08dbc7d9078239ff6f737f))

- Update file structure
  ([`01a864b`](https://github.com/t03i/FlatProt/commit/01a864bfc1c42573b9dc66572022d9b1a63e730b))

### Code Style

- Add non-element drawing
  ([`f452567`](https://github.com/t03i/FlatProt/commit/f4525676f9491c61f2aa12cd64721d4464229c72))

- Add rounded linecaps
  ([`2881a6e`](https://github.com/t03i/FlatProt/commit/2881a6ed0ce9ba3d69d8b7497652df97cdbb0107))

- Make helix connect in the center
  ([`1543288`](https://github.com/t03i/FlatProt/commit/1543288e1d2d13b6cc3459bf799816628657cb32))

### Continuous Integration

- Add pre commit hooks
  ([`b1b4a0b`](https://github.com/t03i/FlatProt/commit/b1b4a0ba46fca9904613ad62830fee56dba936cf))

- Add pytest to pull requests
  ([`3718495`](https://github.com/t03i/FlatProt/commit/3718495cdda73641cbe69f1c0c205f91ed445b70))

- Add releaserc to fix semantic release
  ([`abbc2a5`](https://github.com/t03i/FlatProt/commit/abbc2a50ecb8e3416040a1d43388591b62627cd4))

- Fix installation issues psr
  ([`39ba9d3`](https://github.com/t03i/FlatProt/commit/39ba9d3d47fa3ce0f610d9cbc6ddb00e0f61ce42))

- Update commit parser
  ([`1ed7142`](https://github.com/t03i/FlatProt/commit/1ed714258230382528ad1bed172e4e233ab28db1))

- Update semantic release
  ([`8fc028a`](https://github.com/t03i/FlatProt/commit/8fc028a0ab871ee27608de496bd8924c4b6cb510))

### Documentation

- Add GitHub templates
  ([`f684626`](https://github.com/t03i/FlatProt/commit/f684626a3f809416d2bc6d4802ef385c7dd41667))

- Allow blank issues during pre-release
  ([`40825d4`](https://github.com/t03i/FlatProt/commit/40825d4f38a53e53530c1ba81d888b8bfcd2f535))

- Switch to mkdocs for easier maintenance and editing
  ([`db3235b`](https://github.com/t03i/FlatProt/commit/db3235b5475033829036634f921578935ed09fcc))

### Features

- Add annotation implementation
  ([`6e6b8de`](https://github.com/t03i/FlatProt/commit/6e6b8de364f12e39882db2af0585f1c42902044f))

- Add basic alignment code
  ([`8bbd225`](https://github.com/t03i/FlatProt/commit/8bbd2259dda76b4e0000107e3f181e95f534a1ff))

- Add dssp parser
  ([`4835181`](https://github.com/t03i/FlatProt/commit/4835181ad4e264b093dfa2a404b9051b2577df22))

- Add gap filling with coils
  ([`c2c35ea`](https://github.com/t03i/FlatProt/commit/c2c35eadf6089d3d0c071702b0a3d3f46c8eded2))

- Add projection layer
  ([`7d3f6f2`](https://github.com/t03i/FlatProt/commit/7d3f6f248191ba7ae109ddee9cc81df402826da1))

- Implement custom dssp parser
  ([`0df205a`](https://github.com/t03i/FlatProt/commit/0df205aa9e5eb8d19abcaf049b1d9abffa7805f1))

### Refactoring

- Adapt example to new scene composition
  ([`cafbdc1`](https://github.com/t03i/FlatProt/commit/cafbdc1a1618226ccfcf87b08faf68e2717e75b6))

- Add base elements
  ([`6495a0e`](https://github.com/t03i/FlatProt/commit/6495a0e2a43957b904ebd4befae11a99f96266e3))

- Add basic circle annotation
  ([`af7ba84`](https://github.com/t03i/FlatProt/commit/af7ba849298d9a3a6e160c0e1b1376917ea58fa7))

- Add combined rotation
  ([`5081205`](https://github.com/t03i/FlatProt/commit/50812053314264d099d3cdd0720e188395273ca6))

- Add core entities
  ([`e516fd0`](https://github.com/t03i/FlatProt/commit/e516fd0d5b3aa280842d9ad691fdef2682daa69b))

- Add core exports
  ([`b1c32ca`](https://github.com/t03i/FlatProt/commit/b1c32cad4be906bf8520d885500371f5d0ecef7c))

- Add core module exports
  ([`5f69c98`](https://github.com/t03i/FlatProt/commit/5f69c980704505fe8a9b02acac729a69ec53bb6c))

- Add h5py
  ([`2c9a640`](https://github.com/t03i/FlatProt/commit/2c9a640dff77237c01e825028bf878ee2a017ca5))

- Add index resolution
  ([`da02ef1`](https://github.com/t03i/FlatProt/commit/da02ef1c4d025bc54f92a1a19b04c45f1db69441))

- Add pdb parsing from biopython
  ([`26d51f0`](https://github.com/t03i/FlatProt/commit/26d51f05855a406ce0e05202b82da9adb26d921f))

- Add separate projection abstraction
  ([`09dc118`](https://github.com/t03i/FlatProt/commit/09dc11873e4c8185e0afbd803a2f5fc7180f9dd8))

- Add sorting by z-coordinate
  ([`2ad666d`](https://github.com/t03i/FlatProt/commit/2ad666d49450c668bd8134872686e2a888cffb70))

- Add structure for visualization layer
  ([`8681035`](https://github.com/t03i/FlatProt/commit/868103574c2c45f179b20a47fccee3beffff8dae))

- Add svg rendering
  ([`493c39a`](https://github.com/t03i/FlatProt/commit/493c39a5b6736faf089c660683a247d411fbf73b))

- Change chain to subclass of structure_component
  ([`7545d49`](https://github.com/t03i/FlatProt/commit/7545d49dd64e8717e9c5a03465bf6461f2f61ca4))

- Change display coordinate name to be more fitting
  ([`a4f06fc`](https://github.com/t03i/FlatProt/commit/a4f06fc2bc471ea5424306cfa46f6e3d235e8d81))

- Change name for more clarity
  ([`5c83e21`](https://github.com/t03i/FlatProt/commit/5c83e21aa6f1650a6b48e9703664a4ccc839449d))

- Change theme
  ([`edf76d4`](https://github.com/t03i/FlatProt/commit/edf76d4c17591cbb4587b553330bae8f8fbb06d1))

- Change to python 3.12 to be compatible with biopython
  ([`44a45c0`](https://github.com/t03i/FlatProt/commit/44a45c00d534bb971c245f7ce846ec2e2d8da6b5))

- Change transformer name
  ([`7290bff`](https://github.com/t03i/FlatProt/commit/7290bff9326081760317eb97626b21d34840dd21))

- Cleanup dependencies
  ([`6fc9b55`](https://github.com/t03i/FlatProt/commit/6fc9b553a4f7152461c3da041870d5fd0dbd85e1))

- Coordinate responsibility stays in the manager
  ([`b590c55`](https://github.com/t03i/FlatProt/commit/b590c55195305c87bea3662615939d1182d30cfb))

- Create core structure model
  ([`a739409`](https://github.com/t03i/FlatProt/commit/a739409fe1b8f58bc1782b181657ee342eae8965))

- Export StructureComponent
  ([`9de5a18`](https://github.com/t03i/FlatProt/commit/9de5a180477dd7b5e4bd392c95409d3cb0bfbe1e))

- Fix coil working
  ([`f1466e2`](https://github.com/t03i/FlatProt/commit/f1466e2c0efeeaba6e2bfb8186776715619a4a85))

- Fix drawing
  ([`862543b`](https://github.com/t03i/FlatProt/commit/862543bb0d94a307bc5f811ef16abb10443b268a))

- Fix import
  ([`2434f86`](https://github.com/t03i/FlatProt/commit/2434f867093005109a819348efb67a2210c8d3cd))

- Fix issues and extend tests
  ([`3fb9119`](https://github.com/t03i/FlatProt/commit/3fb91191028ae1c3ba6e3c9a5a85854dfe6302b7))

- Helix path zig-zag pattern
  ([`487fc55`](https://github.com/t03i/FlatProt/commit/487fc5585dd9cbb57adc5ad784684eaf486b821b))

- Improve .cif parsing to handle secondary structure
  ([`7dcf4c0`](https://github.com/t03i/FlatProt/commit/7dcf4c005e27db90f14feb14ca3f1516d1d2a04f))

- Improve naming
  ([`375578d`](https://github.com/t03i/FlatProt/commit/375578df897dcffd6880ac1afcdb4ca82fe00feb))

- Improve scaling
  ([`6af118c`](https://github.com/t03i/FlatProt/commit/6af118cf1168193e91a955410b029fdf40d0eda0))

- Improve styling
  ([`3ab662a`](https://github.com/t03i/FlatProt/commit/3ab662a2bd1ba0946eb3f804f0f7db6a4f389100))

- Improve styling
  ([`960006e`](https://github.com/t03i/FlatProt/commit/960006e0647a16862720acfa3e63db1dec9b307d))

- Improved example visualization
  ([`3f9e801`](https://github.com/t03i/FlatProt/commit/3f9e8010ce454546848158853f26fdc82f7a1408))

- Integrate style for annotation drawing
  ([`227b0b2`](https://github.com/t03i/FlatProt/commit/227b0b2195e5998c83d9ebe337a036b895918a41))

- Keep residue information
  ([`f0691da`](https://github.com/t03i/FlatProt/commit/f0691da714b1abcc1d2d8aeef594f9a6f9ce19b7))

- Make annotation work generally
  ([`0b8e9e2`](https://github.com/t03i/FlatProt/commit/0b8e9e2245dc7e0ed6992dc29eb6cf08e17f1005))

- Move coordinate view into secondary structure
  ([`58a2d28`](https://github.com/t03i/FlatProt/commit/58a2d28252dfcc5115ae44be2bdeb9c136303a7c))

- Move old code to subfolder
  ([`8c6aa03`](https://github.com/t03i/FlatProt/commit/8c6aa03fe17014c65952b9927d1c04263be3531a))

- Project structure and update dependencies
  ([`42b0b6e`](https://github.com/t03i/FlatProt/commit/42b0b6e810ff683da82362638a6c880a51870bb6))

- Expanded .gitignore to include additional Python and IDE files. - Added .python-version file to
  specify Python version 3.13. - Updated pyproject.toml to reflect new project version 0.1.0 and
  modified dependencies. - Removed requirements.txt as dependencies are now managed via
  pyproject.toml. - Introduced new FlatProt.code-workspace for VSCode settings. - Added new data
  files for SF_database and flexUSER_database. - Deleted old distribution files and documentation
  artifacts. - Created initial structure for the FlatProt package with main functionalities and
  utilities

Note docker is broken and needs fixing.

- Pydantic classes for style management
  ([`97fff74`](https://github.com/t03i/FlatProt/commit/97fff7445be8f1d64ba09d944d0d9b77ac62b7b8))

- Remove biopython
  ([`9a748ec`](https://github.com/t03i/FlatProt/commit/9a748ecbab39f1b19bb0ccb4e34bc092be0cf135))

- Remove code duplication for transformation
  ([`2a9d802`](https://github.com/t03i/FlatProt/commit/2a9d802847520ea2a5a468595ce4fdac8af57e42))

- Remove debug output
  ([`07ac5a1`](https://github.com/t03i/FlatProt/commit/07ac5a1d1fe92a7bfba4cd3eb3ca832dc22824cb))

- Remove debug output
  ([`abe1285`](https://github.com/t03i/FlatProt/commit/abe1285cbb9ebc60f24d942f7adabed35c565745))

- Remove explicit structure from projection
  ([`0833ed2`](https://github.com/t03i/FlatProt/commit/0833ed21de3b871414deda57ca754a5436a293d8))

- Remove obsolete data
  ([`5259ce3`](https://github.com/t03i/FlatProt/commit/5259ce3e0f382a83f9d583550bbd2c1407d08d38))

- Remove old code
  ([`485c03d`](https://github.com/t03i/FlatProt/commit/485c03da748a62c4b64f5e892221a9f0de010cc2))

- Remove state from visualization
  ([`0857ed9`](https://github.com/t03i/FlatProt/commit/0857ed95c579a07424880b7cbcec7bb1605c50b4))

- Remove unnecessary types
  ([`fcbb233`](https://github.com/t03i/FlatProt/commit/fcbb2333b6cd3d8946fdf03e81b4ca559b73b1b2))

- Rename ProjectionMatrix to TransformationMatrix
  ([`a5ced0b`](https://github.com/t03i/FlatProt/commit/a5ced0b93c95c9b0dd3c410b0072149a75ba694a))

- Restructure scene for better element management
  ([`eaa2be0`](https://github.com/t03i/FlatProt/commit/eaa2be09de26e606e3387d61525528496a852200))

Scene manages elements in a flat and treelike structure to allow coordinate mapping and tree
  construction.

- Secondary structure owns all data as view
  ([`d9abbdb`](https://github.com/t03i/FlatProt/commit/d9abbdb2e615198fb4ab8edb53d13a5d38d33fa7))

- Separate coordinates from drawing with scene object
  ([`eed0012`](https://github.com/t03i/FlatProt/commit/eed0012e4c290a4e8e08cede4a96a3c7b6d96aa5))

- Separate style from scene
  ([`2596feb`](https://github.com/t03i/FlatProt/commit/2596feb50a1d5b4e115cbf1b5dd6fddec7d1362f))

- Sheet visualization close to original
  ([`a3e18b7`](https://github.com/t03i/FlatProt/commit/a3e18b70a4289f22830b0dcb1008c35d058d5fd8))

- Simplify protein IO
  ([`8bb6b55`](https://github.com/t03i/FlatProt/commit/8bb6b5510e28b46d8a21dbb55b3e4ce97ef36675))

- Simplify smoothing algorithm
  ([`87f9ca5`](https://github.com/t03i/FlatProt/commit/87f9ca5bee5929aab30c361118904be14913a99e))

- Structure to core
  ([`af371db`](https://github.com/t03i/FlatProt/commit/af371db9fc5f011e5f1dc2fbc9741f75385eda26))
