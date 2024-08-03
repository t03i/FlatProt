import typer
from .main import check_pdb_path,get_best_rotation,db_get_ID_info,db_set_SF_pymol_rot,create_2DSVG_from_pdb
from .utils import count_residues,add_header_to_predicted_pdb,extract_sequence_from_pdb,get_pdb_files_for_id_list,format_domain_annotation_file_chainsaw_discMerge,format_domain_annotation_file_chainsaw_discSplit

app = typer.Typer()

app.command()(check_pdb_path)
app.command()(count_residues)
app.command()(get_best_rotation)
app.command()(db_get_ID_info)
app.command()(db_set_SF_pymol_rot)
app.command()(format_domain_annotation_file_chainsaw_discSplit)
app.command()(format_domain_annotation_file_chainsaw_discMerge)
app.command()(create_2DSVG_from_pdb)
app.command()(add_header_to_predicted_pdb)
app.command()(extract_sequence_from_pdb)
app.command()(get_pdb_files_for_id_list)

if __name__ == "__main__":
    app()