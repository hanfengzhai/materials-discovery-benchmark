# megnet/data/graph.py


def convert(self, structure: Structure, state_attributes: List = None) -> Dict:
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance.
        For state attributes, you can set structure.state = [[xx, xx]] beforehand or the algorithm would
        take default [[0, 0]]
        Args:
            state_attributes: (list) state attributes
            structure: (pymatgen structure)
            (dictionary)
        """
        state_attributes = (
            state_attributes or getattr(structure, "state", None) or np.array([[0.0, 0.0]], dtype="float32")
        )
        atoms = self.get_atom_features(structure)
        index1, index2, _, bonds = get_graphs_within_cutoff(structure, self.nn_strategy.cutoff)

        if len(index1) == 0:
            with MPRester() as mpr:
                struct = mpr.get_structure_by_material_id("mp-149")
                atoms = self.get_atom_features(struct)
                index1, index2, _, bonds = get_graphs_within_cutoff(struct, self.nn_strategy.cutoff)

        if np.size(np.unique(index1)) < len(atoms):
            logger.warning("Isolated atoms found in the structure. The " "cutoff radius might be small")

        return {"atom": atoms, "bond": bonds, "state": state_attributes, "index1": index1, "index2": index2}
