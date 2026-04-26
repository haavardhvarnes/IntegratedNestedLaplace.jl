module INLASpatial

using LinearAlgebra
using SparseArrays
using Meshes
using Unitful
using INLACore

export build_mesh, spde_matrices, spde_precision

"""
    spde_precision(C, G, kappa, tau)

Construct the precision matrix for a Matérn GMRF from the mass (C) and stiffness (G) matrices.
Formula: Q = tau * (kappa^2 * C + G) * inv(C_diag) * (kappa^2 * C + G)
"""
function spde_precision(C::SparseMatrixCSC{T}, G::SparseMatrixCSC{T}, kappa::T, tau::T) where T
    # Use diagonal approximation
    C_diag_vals = vec(sum(C, dims=2))
    inv_C_diag = spdiagm(0 => one(T) ./ C_diag_vals)
    
    K = kappa^2 * C + G
    Q = tau * (K * inv_C_diag * K)
    return Q
end

"""
    build_mesh(coords)

Create a 2D Delaunay triangulation from a set of points.
"""
function build_mesh(coords::Vector{<:Tuple{Real, Real}})
    points = [Point(c...) for c in coords]
    pset = PointSet(points)
    # tesselate(pset, DelaunayTesselation()) creates a mesh from all points
    mesh = tesselate(pset, DelaunayTesselation())
    return mesh
end

"""
    spde_matrices(mesh)

Compute the Finite Element mass matrix (C) and stiffness matrix (G) 
for the SPDE discretization on the given mesh.
"""
function spde_matrices(mesh)
    n = nvertices(mesh)
    topo = topology(mesh)
    nelem = nelements(mesh)
    
    C = spzeros(n, n)
    G = spzeros(n, n)
    
    for k in 1:nelem
        cell = element(mesh, k)
        v_indices = indices(element(topo, k))
        
        # Coordinates of vertices without units
        verts = [ustrip.(to(vertex(mesh, i))) for i in v_indices]
        
        # Area of triangle
        area = ustrip(measure(cell))
        
        # Element Mass Matrix (C_e)
        C_e = (area / 12.0) * [2.0 1.0 1.0; 1.0 2.0 1.0; 1.0 1.0 2.0]
        
        # Element Stiffness Matrix (G_e)
        M = [ones(3) [v[1] for v in verts] [v[2] for v in verts]]
        # grads should be a 3x2 matrix where each row is the gradient of a basis function
        # The coefficients (a_i, b_i, c_i) are the i-th column of inv(M)
        # So grad(phi_i) = (inv(M)[2, i], inv(M)[3, i])
        grads = (inv(M)[2:3, :])'
        G_e = area * (grads * grads')
        
        # Assembly
        for i in 1:3
            for j in 1:3
                C[v_indices[i], v_indices[j]] += C_e[i, j]
                G[v_indices[i], v_indices[j]] += G_e[i, j]
            end
        end
    end
    
    return C, G
end

end # module
