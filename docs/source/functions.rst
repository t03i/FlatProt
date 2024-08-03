Usable Methods:
===============

All methods can be called directly from the package without using main./utils. as prefix.
If an error caused by DSSP occurs you might want to delete the DBREF row in the input pdb. This row can cause errors and is unnececary for the DSSP functionality.
The error causing location in the input pdb is often given in the system output.

.. autofunction:: main.create_2DSVG_from_pdb 
.. autofunction:: main.db_get_SF_info
.. autofunction:: main.db_set_SF_pymol_rot
.. autofunction:: main.get_best_rotation
.. autofunction:: main.check_pdb_path

.. autofunction:: utils.format_domain_annotation_file_chainsaw_discMerge
.. autofunction:: utils.format_domain_annotation_file_chainsaw_discSplit
.. autofunction:: utils.add_header_to_predicted_pdb
.. autofunction:: utils.add_header_to_pdb_dir
.. autofunction:: utils.extract_sequence_from_pdb
.. autofunction:: utils.get_pdb_files_for_id_list
