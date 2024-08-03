import svgwrite
import re
import os

from .Graphical import *
from .Main_functions import *
from .Structure_Database import *
from datetime import datetime
import xml.etree.ElementTree as ET
from .utils import check_pdb_files, read_in_foldseek_cluster_tsv,check_and_delete_directory,clear_dir

def check_pdb_path(pdb_file):
    """
    This method checks a path for a existing pdb-file with including a header (necessary for DSSP usage)

    Args:

    - pdb_file (str): Path to file that should be checked

    Returns:

    - the protein id from the file name if file is in correct form. Otherwise None
    
    """
    match = re.search(r'([a-zA-Z0-9]+)\.pdb$', pdb_file)
    #TODO check for header
    if match:
        base_name = os.path.basename(pdb_file)
        file_id = os.path.splitext(base_name)[0]
        with open(pdb_file, 'r') as input_file:
            pdb_content = input_file.read()
        if pdb_content.startswith("HEADER"):
            print("pdb-file with HEADER found!")
            return file_id
        else:
            print("pdb-file withput HEADER found")
            return None
    else:
        print(f"no pdb-file was found at: \"{pdb_file}\"")
        return None  

def visualize(dwg,protein,vis_type, AS_annotation, mark_endings, simple_helix, cysteins, simple_coil_avg, show_lddt_col,show_lddt_text):
    if len(protein.secondary_structures) == 0:
        return
    general_opacity=0.9
    only_path = False
    if(vis_type=='only-path'):
        #protein.get_protein_ordered_vis_objects(1, mark_endings)
        protein.draw_simplified_path(dwg,simple_coil_avg,general_opacity)

        only_path=True
    
    elif vis_type=='simple-coil':
        protein.get_protein_ordered_vis_objects(simple_coil_avg, mark_endings)
        # vis non-coil ss noraml but connect by simplifying coil structure
        visualize_ordered_elements(dwg,protein.ordered_vis_elements, simple_helix, general_opacity, cysteins, show_lddt_col,show_lddt_text)

    elif vis_type=='normal':
        protein.get_protein_ordered_vis_objects(1, mark_endings)
        visualize_ordered_elements(dwg,protein.ordered_vis_elements, simple_helix, general_opacity=general_opacity,cysteins=cysteins, show_lddt_col=show_lddt_col,show_lddt_text=show_lddt_text)
    
    elif vis_type=='special':
        protein.get_protein_ordered_vis_objects(1,mark_endings)
        #create_testing_vis(dwg,ss_objects=protein.secondary_structures)
        visualize_ordered_elements(dwg,protein.ordered_vis_elements, simple_helix, general_opacity=general_opacity,cysteins=cysteins, show_lddt_col=show_lddt_col,show_lddt_text=show_lddt_text, no_coil=True, shadow_col=False)
    
    elif vis_type=='fruchtermann':
        protein.get_protein_ordered_vis_objects(1)
        # vis of non coil ss segments that are pushe apart using Fruchtermann Reingold layout connected with simple lines
        do_fruchtermann_reingold_layout(protein.secondary_structures, k=0.5, iter=50)
        protein.scale_shift_coords(scale=1,shift=20, make_positive=True) #make everything positive
        visualize_ordered_elements(dwg,protein.ordered_vis_elements, simple_helix,general_opacity=general_opacity, cysteins=cysteins, show_lddt_col=show_lddt_col,show_lddt_text=show_lddt_text)
    
    else:
        raise ValueError("Error: Please provide a valid vis_type!")
        # additional:

    if AS_annotation:
        add_residue_annotations(dwg, protein.secondary_structures,only_path)

def get_best_rotation(pdb_file):
    """
    Calculates the best rotation for the input protein. The best rotation is based on maximising the area of content of the 2D representation, 
    minimising the number of overlapping non-coil secondary structures and minimising the depth (z range) of the picture.

    Args:

    - pdb_file (str): Path to pdb file the best rotation should be calculated for.

    Returns:

    - The rotation matrix that can be used to rotate the input protein in the best found orientation
    """
    file_id = check_pdb_path(pdb_file)
    protein = Protein()
    pdb_element = protein.parse_pdb(pdb_file)
    protein.get_secondary_structure(pdb_element,pdb_file)
    
    #rotate protein
    protein.find_best_view_angle()
    print()
    print("Best found rotation: ")
    print(protein.best_rotation)
    return protein.best_rotation

def db_get_ID_info(id:int,user_db:bool = False):
    """
    User can input a SCOP SF and get information on the SF in the database. 

    Args:

    - SF_numner(str): SCOP Superfamily identifier for that the information should be returned

    Returns:

    - The representative score, the protein-representative, the fixed rotation, and the fixed rotation type of the family
    """

    db = Structure_Database(None,None,user_db)
    return db.get_SF_info(id)

def db_set_SF_pymol_rot(id: str,pymol_output, user_db:bool = False):
    """
    Can be used for setting a wanted fixed rotation for a specific Superfamily in the database. 
    Proteins matched to this SF will now be rotated using the new rotation matrix when using family_vis = True.
    The user can first rotate the family representative in pymol and then save the pymol rotation in the database. Every protein will than be rotated that way.
    
    Args:

    - id (str): SCOP SuperFamily number that`s fixed rotation matrix should be changend.
    - pymol_output (str): Pymols get_view() output matrix. Contains rotation matrix
    
    Returns:

    - returns nothing but changes the db entry of the given id. 
    """
    db= Structure_Database(None,None, user_db)
    rot_matrix = transform_pymol_out_to_UT(pymol_output)[0]
    db.set_manual_id_rotation(id, rot_matrix)

def create_USERflex_db(pdb_dir, name_id_mapping_csv, foldseek_executable):
    """
    Method for creating a database using an input directory of proteins (pdb_format) and a mapping file that maps the proteins to ids. This database can then be used for protein visualization by setting user_db=True in the create_2DSVG_from_pdb funcion.

    Args:\n\n
    
    - pdb_dir (str): Path to the directory of proteins in pdb-format.
    - name_id_mapping_csv (str): Path to protein-id-mapping CSV file. The ids must be Integers.
    - foldseek_executable (str): Path to the foldseek executable. Is used to create foldseek-database out of the protein structures.
    
    Creates a user-specified database that can then be used for visualization.
    """
    structure_database_obj = Structure_Database(foldseek_executable,None, True)
    # create user-specified databse for comparing input proteins to. needs
    # needs pdb_dir and matching name -> id list for tsv info
    valid_pdbs, invalid_pdbs = check_pdb_files(pdb_dir)
    if len(invalid_pdbs)>0:
        print("\n\033[91m" +"USER-database creation error: invalid PDBs!\n")
        print(f"valid pdbs: {len(valid_pdbs)}\ninvalid pdbs: {len(invalid_pdbs)} ({invalid_pdbs})\nPlease redo database creation with only valid pdb files. (parseable by PDBParser from Bio.PDB)"+"\033[0m\n")
        return
    info_check = structure_database_obj.check_create_USERdb_info_table(name_id_mapping_csv,structure_database_obj.db_info_tsv)
    if info_check == None:
        print("\n\033[91m" +"USER-database creation error: name-id mapping CSV incorrect format!"+"\033[0m\n")
        print("Check the documentation for the needed format:")
        print("format example:\n--------------\nID,NAME\nid_1,name_1\nid_2,name_2\nid_3,name_3\n--------------")
        return

    structure_database_obj.create_USER_PDB_foldseek_index(pdb_dir)
    structure_database_obj.last_altered = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def create_2DSVG_from_pdb(pdb_file:str, result_dir:str, tmp_dir:str, family_vis:bool=True, fam_aligned_splitting:bool = True, drop_family_prob:float = 0.5,foldseek_executable:str="/foldseek/bin/foldseek",user_db:bool=False,fixed_id:int=None,id_info:bool=True ,domain_splitted:bool=False, domain_annotation_file:str=None, domain_hull:bool=True, visualisation_type:str ="normal", 
                    cysteins:bool=True, as_annotation:bool=False, mark_endings:bool=True, simple_helix:bool=True, show_lddt_col:bool=False,show_lddt_text:bool=False, find_best_rot_step:int = 30, simple_coil_avg:int=10, chain_info:bool=True, no_shifting=False):
    """
    Main method of the package for creating 2D visualisations of a protein in form of a SVG file. The user can decide between different visualisation options.
    
    Args:\n\n

    - pdb_file (str): Path to pdb file the visualisation file is generated on. (Required)\n
    - result_dir (str): Path to dir, where the output file is saved (file name is automatically set based on input file name). (Required)\n
    - tmp_dir (str): Path to dir, where temporary files needed for analyis (e.g. foldseek) and visualisations are saved. (Required)\n
    - foldseek_executable (str): Path to foldseek executable (will be used for family alignment). Default is "foldseek/bin/foldseek"\n
    
    - family_vis (bool): If True, enables family-wise visualization, uses SCOP SF database with calculated representatives. Default is True.\n
    - fam_aligned_splitting (bool): If True, the protein is split into SF-aligned (is rotated based on this segment) and not-aligned parts. THey are connected with a dashed line. Default is True.\n
    - drop_family_prob (float): Allows the program to drop a chosen SF if the FoldSeek probability is smaller than given cut-off. In this case the protein rotation is determined using the implemented "find_best_view_angle" method. Default is 0.5. \n
    - show_lddt_col (bool): LDDT scores from FoldSeek alignment to best matching SF is shown per residue as colorscale (magenta). Default is False. \n
    - show_lddt_text (bool): LDDT scores from FoldSeek alignment to best matching SF is shown per residue. Default is False. \n
    - fixed_id (int): Fixed id that will be used for the protein (/ for every domain in the protein) (to distant proteins cannot be used for aligning)\n
    - id_info (bool): If True, adds assigned ID number and corresponding foldseek probability to the drawing. Default is True.\n
    - user_db (bool): If True, prot2d uses the user-created database for protein matching. The user-db must be created before with the "create_USERflex_db" function. Default is False for simple usage with SCOP superfamily database\n4
    
    - domain_splitted (bool): If True, protein is split into domains using the provided domain annotation file. Can be used in combination with family_vis which is then applied on each domain seperatly. Default is False.\n
    - domain_annotation_file (str): Path to the domain annotation file. Required if domain_splitted is True.\n
    - domain_hull (bool): If True sourounds domains with smoothed convex hull colored based on the secondary structure composition (R,G,B) <-> (helix,sheet,coil). Default is True\n

    - visualisation_type (string): "only-path", "normal", or "simple-coil". Default is "normal".\n
    - cysteins (bool): If True, includes calculated cystein bonds in the visualisation. Default is True.\n
    - as_annotation (bool): If True, includes AS-annotation. Default is False.\n
    - mark_endings (bool): If True, marks the endings. Default is True.\n
    - simple_helix (bool): If True, helix are represented in a simple way (file size eficient). Default is True.\n
    - find_best_rot_step (int): Is the size of steps per 3D rotation angle (total=3) taken to find the rotatoin showing the most of the input protein. Increasing leads to faster runtime but worse visualisations. Default is 30.\n
    - simple_coil_avg (int): Coil structures will be summarised together. e.g. 10 means that every 10 coil-residues will be averaged and treated as one point. Bigger values lead to higher simplification. Is only used when "simple-coil" or "only-path" is used. Default is 10\n
    - chain_info (bool): If true and multi-chain protein structure is given: adds chain annotations (from pdb) in the visualizations. Default is True.\n

    Returns: The path to the created SVG-file. \n

    - Creates a SVG file containing the 2D visualisation of the input protein in the given result_dir.
    """
    ############## Validate arguments ##############
    if domain_splitted and not domain_annotation_file:
        raise ValueError("Domain annotation file is required for domain-split analysis.")
    if visualisation_type=="split-alignment" and not family_vis:
        raise ValueError("Alignment split option can only be used when doing the family visualisation (aligned part is used for splitting).")
    valid_vis_types = ["only-path","normal","simple-coil","special"]
    if visualisation_type not in valid_vis_types:
        raise ValueError(f'"{visualisation_type}" is not a valid visualisation type. Please use one of the following: {valid_vis_types}')
    file_id = check_pdb_path(pdb_file)
    if file_id ==None:
        raise ValueError(f'"{pdb_file}" is not a valid pdb input. Please check the path and the file.')
    #counteract impossible combinations
    if not family_vis:
        fam_aligned_splitting=False
        drop_family_prob=False
    
    structure_database_obj = Structure_Database(foldseek_executable,tmp_dir, user_db)

    chain_pdb_dict = split_pdb_chains(pdb_file,tmp_dir)
    print(f"{len(chain_pdb_dict)} chains were found in the input pdb and will be visualized seperatly")
    print(chain_pdb_dict)
    chain_prots = []
    all_chain_domain_prots =[]
    for chain_id,chain_pdb in chain_pdb_dict.items():
        ############## 1) Split into domains if used ##############
        if domain_splitted:
            #TODO adapt domain splitting per chain...(chainsaw reformatting changes...)
            domain_files = get_domain_pdbs(chain_pdb,chain_id,domain_annotation_file,tmp_dir)
        else:
            domain_files = []
            domain_files.append(chain_pdb)

        print(f"{len(domain_files)} domain(s) were found for chain {chain_id} and will be visualized seperatly:")
        if len(domain_files) == 0:
            continue
        ############## 2) Get rotation of protein (family-vis / best rotation) ##############
        dom_proteins = []
        matched_sfs = []
        probs = []
        for dom_file in domain_files:
            print(f"\n### \"{dom_file}\" ###\n")
            if family_vis:
                sf,prob,sf_aligned_pdb_file, aligned_region,lddtfull = structure_database_obj.initial_and_fixed_Sf_rot_region(dom_file,drop_family_prob,fixed_sf=fixed_id, flex_USER_db=user_db)
                if sf_aligned_pdb_file== None:
                    #no mathcing sf (with higher than min prob found) --> do normal vis
                    print("\033[91m"+ "No maching id (with higher prob than min prob) found. Normal roation algo will be used!"+"\033[0m" )
                    add_header_to_pdb(dom_file)
                    dom_prot = Protein()
                    dom_prot.chain = chain_id
                    pdb_element = dom_prot.parse_pdb(dom_file)
                    dom_prot.get_secondary_structure(pdb_element,dom_file)
                    dom_prot.scale_shift_coords(scale=30,x_shift=0,y_shift=0, make_positive=False)
                    dom_prot.find_best_view_angle(step_width = find_best_rot_step)
                    dom_proteins.append(dom_prot)
                    matched_sfs.append("no_SF")
                    probs.append(-1)
                    continue
                    
                add_header_to_pdb(sf_aligned_pdb_file)
                dom_prot = Protein()
                dom_prot.chain = chain_id
                pdb_element = dom_prot.parse_pdb(sf_aligned_pdb_file)
                dom_prot.get_secondary_structure(pdb_element,sf_aligned_pdb_file)
                dom_prot.scale_shift_coords(scale=30,x_shift=0,y_shift=0, make_positive=False)
                dom_prot.add_lddt_to_aligned_residues(aligned_region,lddtfull) if show_lddt_col or show_lddt_text else None
                dom_prot.sf= sf
                matched_sfs.append(sf)
                dom_prot.prob= prob
                probs.append(prob)
                if fam_aligned_splitting:
                    # split protein in 3 segments (aligment-based)
                    front_part,aligned_part,end_part = dom_prot.split_aligned_part(aligned_region)
                    # shift left and right part to the sides and make positive again
                    front_aligned_x_shift = calc_x_overlap_distance_between_prots(front_part,aligned_part) + 200
                    aligned_end_x_shift = calc_x_overlap_distance_between_prots(aligned_part,end_part) + 200
                    end_part.scale_shift_coords(scale=1,x_shift=aligned_end_x_shift,y_shift=0,make_positive=False)
                    front_part.scale_shift_coords(scale=1,x_shift=-front_aligned_x_shift,y_shift=0,make_positive=False)
            else:
                # manually find best rotation and continue with that
                add_header_to_pdb(dom_file)
                dom_prot = Protein()
                dom_prot.chain = chain_id
                pdb_element = dom_prot.parse_pdb(dom_file)
                dom_prot.get_secondary_structure(pdb_element,dom_file)
                dom_prot.print_ss_objects()
                dom_prot.scale_shift_coords(scale=30,x_shift=0,y_shift=0, make_positive=False)
                dom_prot.find_best_view_angle(step_width = find_best_rot_step)
            dom_proteins.append(dom_prot)
     
        # make all coords postive
        chain_residues = [res for dom in dom_proteins for res in dom.residues]
        chain_prot = Protein()
        chain_prot.residues = chain_residues
        if not no_shifting:
            chain_prot.scale_shift_coords(scale=1,x_shift=0,y_shift=0,make_positive=True)
            #shift domains to be in linear line:
            shift_domains_in_x_line(dom_proteins, 100)
        chain_prots.append(chain_prot)
        all_chain_domain_prots.append(dom_proteins)
    
    full_residues = [res for chain_prot in chain_prots for res in chain_prot.residues]
    full_prot = Protein()
    full_prot.residues = full_residues
    #shift for border space
    if not no_shifting:
        full_prot.scale_shift_coords(scale=1,x_shift=100,y_shift=100,make_positive=False)
        shift_chains_in_y_line(chain_prots=chain_prots,chain_gap=140)
    else:
        full_prot.scale_shift_coords(scale=1,x_shift=100,y_shift=1100,make_positive=False)
    ############## 3) & 4) Create visualisation as wanted: only-path / normal / simple-coil (+ fam_aligned_splitting) ##############
    viewbox = calculate_viewbox(full_prot.residues,300) 
    #viewbox = f"{0} {0} {3000} {3000}"
    result_file_path = result_dir+"/"+file_id+'_'+visualisation_type+'_familyVis_'+str(family_vis)+'_simpleHelix_'+str(simple_helix)+'_vis.svg'
    dwg = svgwrite.Drawing(result_file_path, viewBox=viewbox)
    
    print("\n### Starting visualizing of protein (domains): ###")   
    for chain_doms in all_chain_domain_prots:
        last_dom=None
        for dom_prot in chain_doms:
            if fam_aligned_splitting and dom_prot.fam_aligned_parts!=None :
                front_part, aligned_part, end_part = dom_prot.fam_aligned_parts
                visualize(dwg,front_part,visualisation_type, as_annotation, mark_endings, simple_helix, cysteins, simple_coil_avg, show_lddt_col,show_lddt_text)
                visualize(dwg,aligned_part,visualisation_type, as_annotation, mark_endings, simple_helix, cysteins,simple_coil_avg, show_lddt_col,show_lddt_text)
                visualize(dwg,end_part,visualisation_type, as_annotation, mark_endings, simple_helix, cysteins,simple_coil_avg, show_lddt_col,show_lddt_text)
                add_dashed_line_between_proteins(dwg,front_part,aligned_part,end_part)
            else:
                visualize(dwg,dom_prot,visualisation_type, as_annotation, mark_endings, simple_helix, cysteins,simple_coil_avg, show_lddt_col,show_lddt_text)
            
            if domain_splitted:
                dom_prot.add_hull(dwg,dom_prot.get_hull_color(), opacity=0.25) if domain_hull else None
                dwg.add(last_dom.connect_to_protein_dashline(dom_prot)) if last_dom != None else None
            if id_info:
                dom_prot.add_sf_info(dwg)
            last_dom=dom_prot
        if chain_info and len(all_chain_domain_prots)>1:
            chain_doms[0].add_chain_info(dwg)

            
    print(f"\n### Visualization done! SVG file was created at \"{result_file_path}\" ###")
    dwg.save()
    return result_file_path, matched_sfs,probs

def create_family_overlay(pdb_dir:str,overlay_result_file:str,tmp_dir:str,foldseek_executable:str=None, outlier_cutoff:int=1, keep_tmp_data:bool=False, foldseek_c='0.9',foldseek_min_seq_id='0.5'):
    YELLOW = '\033[93m'
    RESET = '\033[0m'

    # directory management
    process_tmp_dir = os.path.join(tmp_dir,"process_tmp")
    svg_tmp_dir = os.path.join(tmp_dir,"SVGs_tmp")
    os.mkdir(process_tmp_dir) if not os.path.exists(process_tmp_dir) else None
    os.mkdir(svg_tmp_dir)if not os.path.exists(svg_tmp_dir) else None
    cluster_result_files = os.path.join(process_tmp_dir, "res")
    if not check_and_delete_directory(svg_tmp_dir):
        exit()
    if not check_and_delete_directory(process_tmp_dir):
        exit()
    
    # use Foldseek for family clustering
    subprocess.run([
    foldseek_executable, 'easy-cluster', pdb_dir, cluster_result_files, process_tmp_dir, '-c', foldseek_c,'--min-seq-id',foldseek_min_seq_id
    ])
    cluster_tsv_file = f'{cluster_result_files}_cluster.tsv'
    cluster_dict = read_in_foldseek_cluster_tsv(cluster_tsv_file)

    # remove outliers below cutoff
    og_cluster_len = len(cluster_dict)
    cluster_dict = {key: value for key, value in cluster_dict.items() if value > outlier_cutoff}
    filtered_clusuter_len = len(cluster_dict)
    print(f"\n{YELLOW}Deleted cluster outliers (<{outlier_cutoff} entries): {og_cluster_len-filtered_clusuter_len} (Settings: coverage: {foldseek_c}, min_seq_id: {foldseek_min_seq_id})")
    user_input = input(f"Do you want to continue visualization with the remaining {filtered_clusuter_len} (out of {og_cluster_len}) cluster-representatives? [Y/N]: {RESET}").strip().upper()
    if user_input != 'Y':
        return
    failed_files=[]
    # create svgs for cluster representatives
    
    for representative in cluster_dict.keys():
        pdb_file = os.path.join(pdb_dir, representative)
        try:
            result_file,sf,p =create_2DSVG_from_pdb(pdb_file=pdb_file,result_dir=svg_tmp_dir,tmp_dir=process_tmp_dir,family_vis=True, fam_aligned_splitting=False, domain_splitted=False, visualisation_type ="simple-coil", 
            cysteins=False, as_annotation=False, mark_endings=False, simple_helix=True, show_lddt_col=False,show_lddt_text=False,
            domain_annotation_file="", drop_family_prob = 0.1,foldseek_executable=foldseek_executable,fixed_id=None, find_best_rot_step=30, simple_coil_avg=10,chain_info=False, id_info=False, no_shifting=True)
        except:
            failed_files.append(pdb_file)
        
        clear_dir(process_tmp_dir) if not keep_tmp_data else None
    if len(failed_files) != 0:
        output_file = os.path.join(process_tmp_dir,"failed_files.txt")
        with open(output_file, 'w') as file:
            for pdb in failed_files:
                file.write(f"{pdb}\n")
    
    # combine SVGs
    opacity_dict = calculate_opacity(cluster_dict)
    merge_svgs(svg_tmp_dir,opacity_dict,overlay_result_file)
    # clear tmp svg dir
    #! clear_dir(svg_tmp_dir) if not keep_tmp_data else None
        

def calculate_opacity(cluster_dict):
    min_opacity = 0.002
    max_opacity = 1

    min_count = min(cluster_dict.values())
    max_count = max(cluster_dict.values())
    total_clusters = len(cluster_dict)
    opacities_dict = {}
    for key, count in cluster_dict.items():
        if max_count == min_count:
            # If all counts are the same, assign max_opacity to avoid division by zero
            opacities_dict[key] = max_opacity
        else:
            # Calculate the normalized opacity
            normalized_count = (count - min_count) / (max_count - min_count)
            # Adjust the opacity based on the total number of clusters
            #opacity = min_opacity + (normalized_count * (max_opacity - min_opacity)) / total_clusters
            opacity = min_opacity + normalized_count * (max_opacity - min_opacity)
            opacities_dict[key] = opacity
    return opacities_dict

def get_max_viewbox(folder_path,svg_files, padding = 50):
    min_x,min_y = float('inf'),float('inf')
    max_x,max_y = float('-inf'),float('-inf')
    # Find min and max values for x and y coordinates in all SVG files
    for file in svg_files:
        file_path = os.path.join(folder_path, file)
        tree = ET.parse(file_path)
        root = tree.getroot()
        for element in root.iter():
            if 'points' in element.attrib:
                points = element.attrib['points'].split()
                for point in points:
                    x, y = map(float, point.split(','))
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
            if 'cx' in element.attrib and 'cy' in element.attrib:
                cx = float(element.attrib['cx'])
                cy = float(element.attrib['cy'])
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
    # add padding
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding
    # Calculate width and height of viewBox based on min and max values
    viewbox_width = max_x - min_x if max_x != float('-inf') else 0
    viewbox_height = max_y - min_y if max_y != float('-inf') else 0

    return min_x,min_y,viewbox_width,viewbox_height

def merge_svgs(folder_path,opacity_dict, output_file):
    svg_files = [f for f in os.listdir(folder_path) if f.endswith('.svg')]  
    # calc viewbox needed to show evertything!
    min_x,min_y,viewbox_width,viewbox_height = get_max_viewbox(folder_path,svg_files)

    # Create combined SVG element with viewBox based on calculated width and height
    combined_svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", viewBox=f"{min_x} {min_y} {viewbox_width} {viewbox_height}")
    defs_combined = ET.SubElement(combined_svg, "defs")
    counter = 0
    # Iterate over SVG files
    for file in svg_files:
        representative = file.replace('_simple-coil_familyVis_True_simpleHelix_True_vis.svg', '.pdb')
        if not representative in opacity_dict.keys():
            continue
        else:    
            opacity = opacity_dict[representative]
        
        #opacity= 0.02
        counter+=1
        file_path = os.path.join(folder_path, file)

        tree = ET.parse(file_path)
        root = tree.getroot()
        defs = root.find('.//defs')
        if defs is not None:
            for elem in list(defs):
                defs_combined.append(elem)
        g = ET.SubElement(combined_svg, "g", opacity=str(opacity))
        for element in list(root):
            if element.tag != 'defs':
                g.append(element)

    # Write the combined SVG to the output file
    tree = ET.ElementTree(combined_svg)
    tree.write(output_file)

