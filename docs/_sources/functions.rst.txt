Main FlatProt Methods:
======================

All methods can be called directly from the package without using main./utils. as prefix.
If an error caused by DSSP occurs you might want to delete the DBREF row in the input pdb. This row can cause errors and is unnececary for the DSSP functionality.
The error causing location in the input pdb is often given in the system output.

.. autofunction:: FlatProt.main.create_2DSVG_from_pdb
.. autofunction:: FlatProt.main.db_get_ID_info
.. autofunction:: FlatProt.main.db_set_ID_pymol_rot
.. autofunction:: FlatProt.main.create_USERflex_db


Additional Methods:
===================

All methods can be called directly from the package without using main./utils. as prefix.
If an error caused by DSSP occurs you might want to delete the DBREF row in the input pdb. This row can cause errors and is unnececary for the DSSP functionality.
The error causing location in the input pdb is often given in the system output.

.. autofunction:: FlatProt.utils.domAnnot_chainsaw_discMerge
.. autofunction:: FlatProt.utils.domAnnot_chainsaw_discSplit
.. autofunction:: FlatProt.main.get_best_rotation
.. autofunction:: FlatProt.main.check_pdb_path
.. autofunction:: FlatProt.utils.add_header_to_predicted_pdb
.. autofunction:: FlatProt.utils.add_header_to_pdb_dir
.. autofunction:: FlatProt.utils.extract_sequence_from_pdb
.. autofunction:: FlatProt.utils.get_pdb_files_for_id_list
