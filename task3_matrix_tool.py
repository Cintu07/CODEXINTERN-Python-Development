import numpy as np
import sys

class MatrixOperationsTool:
    """Interactive Matrix Operations Tool using NumPy"""
    
    def __init__(self):
        self.matrix_a = None
        self.matrix_b = None
    
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "=" * 60)
        print("          MATRIX OPERATIONS TOOL - NUMPY")
        print("=" * 60)
        print("\n1. Input Matrix A")
        print("2. Input Matrix B")
        print("3. Display Matrices")
        print("4. Matrix Addition (A + B)")
        print("5. Matrix Subtraction (A - B)")
        print("6. Matrix Multiplication (A × B)")
        print("7. Transpose of Matrix A")
        print("8. Transpose of Matrix B")
        print("9. Determinant of Matrix A")
        print("10. Determinant of Matrix B")
        print("11. Inverse of Matrix A")
        print("12. Inverse of Matrix B")
        print("13. Scalar Multiplication")
        print("14. Element-wise Multiplication")
        print("15. Matrix Power")
        print("16. Eigenvalues and Eigenvectors")
        print("0. Exit")
        print("=" * 60)
    
    def input_matrix(self, matrix_name):
        """Input a matrix from user"""
        try:
            print(f"\n--- Input {matrix_name} ---")
            rows = int(input(f"Enter number of rows for {matrix_name}: "))
            cols = int(input(f"Enter number of columns for {matrix_name}: "))
            
            print(f"\nEnter elements row by row (space-separated):")
            matrix = []
            for i in range(rows):
                while True:
                    try:
                        row_input = input(f"Row {i+1}: ").strip()
                        row = [float(x) for x in row_input.split()]
                        if len(row) != cols:
                            print(f"Error: Please enter exactly {cols} values")
                            continue
                        matrix.append(row)
                        break
                    except ValueError:
                        print("Error: Please enter valid numbers")
            
            return np.array(matrix)
        
        except ValueError:
            print("Error: Invalid input. Please enter valid numbers.")
            return None
    
    def display_matrix(self, matrix, name):
        """Display a matrix in formatted way"""
        if matrix is None:
            print(f"\n{name} is not defined yet.")
            return
        
        print(f"\n{name} ({matrix.shape[0]}×{matrix.shape[1]}):")
        print("-" * 40)
        for row in matrix:
            print("  ", end="")
            for val in row:
                print(f"{val:8.2f}", end=" ")
            print()
        print("-" * 40)
    
    def matrix_addition(self):
        """Add two matrices"""
        if self.matrix_a is None or self.matrix_b is None:
            print("\nError: Both matrices must be defined first.")
            return
        
        if self.matrix_a.shape != self.matrix_b.shape:
            print(f"\nError: Matrices must have the same dimensions.")
            print(f"Matrix A: {self.matrix_a.shape}, Matrix B: {self.matrix_b.shape}")
            return
        
        result = self.matrix_a + self.matrix_b
        print("\n*** MATRIX ADDITION (A + B) ***")
        self.display_matrix(result, "Result")
    
    def matrix_subtraction(self):
        """Subtract two matrices"""
        if self.matrix_a is None or self.matrix_b is None:
            print("\nError: Both matrices must be defined first.")
            return
        
        if self.matrix_a.shape != self.matrix_b.shape:
            print(f"\nError: Matrices must have the same dimensions.")
            print(f"Matrix A: {self.matrix_a.shape}, Matrix B: {self.matrix_b.shape}")
            return
        
        result = self.matrix_a - self.matrix_b
        print("\n*** MATRIX SUBTRACTION (A - B) ***")
        self.display_matrix(result, "Result")
    
    def matrix_multiplication(self):
        """Multiply two matrices"""
        if self.matrix_a is None or self.matrix_b is None:
            print("\nError: Both matrices must be defined first.")
            return
        
        if self.matrix_a.shape[1] != self.matrix_b.shape[0]:
            print(f"\nError: Number of columns in A must equal rows in B.")
            print(f"Matrix A: {self.matrix_a.shape}, Matrix B: {self.matrix_b.shape}")
            return
        
        result = np.dot(self.matrix_a, self.matrix_b)
        print("\n*** MATRIX MULTIPLICATION (A × B) ***")
        self.display_matrix(result, "Result")
    
    def transpose(self, matrix, name):
        """Calculate transpose of a matrix"""
        if matrix is None:
            print(f"\nError: {name} is not defined yet.")
            return
        
        result = np.transpose(matrix)
        print(f"\n*** TRANSPOSE OF {name} ***")
        self.display_matrix(result, f"{name}ᵀ")
    
    def determinant(self, matrix, name):
        """Calculate determinant of a matrix"""
        if matrix is None:
            print(f"\nError: {name} is not defined yet.")
            return
        
        if matrix.shape[0] != matrix.shape[1]:
            print(f"\nError: Determinant only exists for square matrices.")
            print(f"{name} is {matrix.shape[0]}×{matrix.shape[1]}")
            return
        
        det = np.linalg.det(matrix)
        print(f"\n*** DETERMINANT OF {name} ***")
        print(f"det({name}) = {det:.4f}")
        
        if abs(det) < 1e-10:
            print("Note: Matrix is singular (determinant ≈ 0)")
    
    def inverse(self, matrix, name):
        """Calculate inverse of a matrix"""
        if matrix is None:
            print(f"\nError: {name} is not defined yet.")
            return
        
        if matrix.shape[0] != matrix.shape[1]:
            print(f"\nError: Inverse only exists for square matrices.")
            print(f"{name} is {matrix.shape[0]}×{matrix.shape[1]}")
            return
        
        try:
            det = np.linalg.det(matrix)
            if abs(det) < 1e-10:
                print(f"\nError: {name} is singular (determinant ≈ 0)")
                print("Inverse does not exist.")
                return
            
            inv = np.linalg.inv(matrix)
            print(f"\n*** INVERSE OF {name} ***")
            self.display_matrix(inv, f"{name}⁻¹")
            
            # Verify: A × A⁻¹ = I
            identity = np.dot(matrix, inv)
            print("\nVerification (A × A⁻¹ should be Identity):")
            self.display_matrix(identity, "A × A⁻¹")
            
        except np.linalg.LinAlgError:
            print(f"\nError: {name} is singular and cannot be inverted.")
    
    def scalar_multiplication(self):
        """Multiply matrix by a scalar"""
        print("\n--- Scalar Multiplication ---")
        print("1. Multiply Matrix A by scalar")
        print("2. Multiply Matrix B by scalar")
        
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == '1':
            matrix = self.matrix_a
            name = "Matrix A"
        elif choice == '2':
            matrix = self.matrix_b
            name = "Matrix B"
        else:
            print("Invalid choice.")
            return
        
        if matrix is None:
            print(f"\nError: {name} is not defined yet.")
            return
        
        try:
            scalar = float(input("Enter scalar value: "))
            result = scalar * matrix
            print(f"\n*** SCALAR MULTIPLICATION ({scalar} × {name}) ***")
            self.display_matrix(result, "Result")
        except ValueError:
            print("Error: Invalid scalar value.")
    
    def element_wise_multiplication(self):
        """Element-wise multiplication of two matrices"""
        if self.matrix_a is None or self.matrix_b is None:
            print("\nError: Both matrices must be defined first.")
            return
        
        if self.matrix_a.shape != self.matrix_b.shape:
            print(f"\nError: Matrices must have the same dimensions.")
            print(f"Matrix A: {self.matrix_a.shape}, Matrix B: {self.matrix_b.shape}")
            return
        
        result = self.matrix_a * self.matrix_b
        print("\n*** ELEMENT-WISE MULTIPLICATION (A ⊙ B) ***")
        self.display_matrix(result, "Result")
    
    def matrix_power(self):
        """Calculate matrix power"""
        print("\n--- Matrix Power ---")
        print("1. Power of Matrix A")
        print("2. Power of Matrix B")
        
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == '1':
            matrix = self.matrix_a
            name = "Matrix A"
        elif choice == '2':
            matrix = self.matrix_b
            name = "Matrix B"
        else:
            print("Invalid choice.")
            return
        
        if matrix is None:
            print(f"\nError: {name} is not defined yet.")
            return
        
        if matrix.shape[0] != matrix.shape[1]:
            print(f"\nError: Matrix power only exists for square matrices.")
            return
        
        try:
            power = int(input("Enter power (integer): "))
            result = np.linalg.matrix_power(matrix, power)
            print(f"\n*** {name}^{power} ***")
            self.display_matrix(result, "Result")
        except ValueError:
            print("Error: Invalid power value.")
    
    def eigenvalues_eigenvectors(self):
        """Calculate eigenvalues and eigenvectors"""
        print("\n--- Eigenvalues and Eigenvectors ---")
        print("1. For Matrix A")
        print("2. For Matrix B")
        
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == '1':
            matrix = self.matrix_a
            name = "Matrix A"
        elif choice == '2':
            matrix = self.matrix_b
            name = "Matrix B"
        else:
            print("Invalid choice.")
            return
        
        if matrix is None:
            print(f"\nError: {name} is not defined yet.")
            return
        
        if matrix.shape[0] != matrix.shape[1]:
            print(f"\nError: Eigenvalues only exist for square matrices.")
            return
        
        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            print(f"\n*** EIGENVALUES AND EIGENVECTORS OF {name} ***")
            print("\nEigenvalues:")
            for i, val in enumerate(eigenvalues):
                if np.isreal(val):
                    print(f"  λ{i+1} = {val.real:.4f}")
                else:
                    print(f"  λ{i+1} = {val.real:.4f} + {val.imag:.4f}i")
            
            print("\nEigenvectors (column-wise):")
            self.display_matrix(eigenvectors.real, "Eigenvectors")
        except np.linalg.LinAlgError:
            print("Error: Could not compute eigenvalues/eigenvectors.")
    
    def run(self):
        """Main loop for the tool"""
        print("\n" + "=" * 60)
        print("     WELCOME TO MATRIX OPERATIONS TOOL")
        print("=" * 60)
        
        while True:
            self.display_menu()
            choice = input("\nEnter your choice (0-16): ").strip()
            
            if choice == '0':
                print("\n" + "=" * 60)
                print("Thank you for using Matrix Operations Tool!")
                print("=" * 60)
                sys.exit()
            
            elif choice == '1':
                self.matrix_a = self.input_matrix("Matrix A")
                if self.matrix_a is not None:
                    print("\n✓ Matrix A stored successfully!")
                    self.display_matrix(self.matrix_a, "Matrix A")
            
            elif choice == '2':
                self.matrix_b = self.input_matrix("Matrix B")
                if self.matrix_b is not None:
                    print("\n✓ Matrix B stored successfully!")
                    self.display_matrix(self.matrix_b, "Matrix B")
            
            elif choice == '3':
                print("\n*** CURRENT MATRICES ***")
                self.display_matrix(self.matrix_a, "Matrix A")
                self.display_matrix(self.matrix_b, "Matrix B")
            
            elif choice == '4':
                self.matrix_addition()
            
            elif choice == '5':
                self.matrix_subtraction()
            
            elif choice == '6':
                self.matrix_multiplication()
            
            elif choice == '7':
                self.transpose(self.matrix_a, "Matrix A")
            
            elif choice == '8':
                self.transpose(self.matrix_b, "Matrix B")
            
            elif choice == '9':
                self.determinant(self.matrix_a, "Matrix A")
            
            elif choice == '10':
                self.determinant(self.matrix_b, "Matrix B")
            
            elif choice == '11':
                self.inverse(self.matrix_a, "Matrix A")
            
            elif choice == '12':
                self.inverse(self.matrix_b, "Matrix B")
            
            elif choice == '13':
                self.scalar_multiplication()
            
            elif choice == '14':
                self.element_wise_multiplication()
            
            elif choice == '15':
                self.matrix_power()
            
            elif choice == '16':
                self.eigenvalues_eigenvectors()
            
            else:
                print("\n❌ Invalid choice. Please select 0-16.")
            
            input("\nPress Enter to continue...")


# Example usage with pre-defined matrices for demonstration
def demo_mode():
    """Run a demonstration with sample matrices"""
    print("\n" + "=" * 60)
    print("     DEMO MODE - Matrix Operations Tool")
    print("=" * 60)
    
    # Sample matrices
    A = np.array([[2, 3, 1],
                  [4, 1, 2],
                  [1, 5, 3]])
    
    B = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    C = np.array([[2, 1],
                  [5, 3]])
    
    print("\nSample Matrix A (3×3):")
    print(A)
    
    print("\nSample Matrix B (3×3):")
    print(B)
    
    print("\nSample Matrix C (2×2):")
    print(C)
    
    # Perform various operations
    print("\n" + "=" * 60)
    print("DEMONSTRATION OF OPERATIONS")
    print("=" * 60)
    
    # Addition
    print("\n1. Matrix Addition (A + B):")
    print(A + B)
    
    # Subtraction
    print("\n2. Matrix Subtraction (A - B):")
    print(A - B)
    
    # Multiplication
    print("\n3. Matrix Multiplication (A × B):")
    print(np.dot(A, B))
    
    # Transpose
    print("\n4. Transpose of A:")
    print(np.transpose(A))
    
    # Determinant
    print("\n5. Determinant of A:")
    print(f"det(A) = {np.linalg.det(A):.4f}")
    
    # Inverse of C (using 2×2 matrix as it's easier to verify)
    print("\n6. Inverse of C:")
    print(np.linalg.inv(C))
    
    # Verify inverse
    print("\n   Verification (C × C⁻¹):")
    print(np.dot(C, np.linalg.inv(C)))
    
    # Scalar multiplication
    print("\n7. Scalar Multiplication (3 × A):")
    print(3 * A)
    
    # Element-wise multiplication
    print("\n8. Element-wise Multiplication (A ⊙ B):")
    print(A * B)
    
    # Matrix power
    print("\n9. Matrix Power (C²):")
    print(np.linalg.matrix_power(C, 2))
    
    # Eigenvalues and eigenvectors
    print("\n10. Eigenvalues of C:")
    eigenvalues, eigenvectors = np.linalg.eig(C)
    print(f"λ1 = {eigenvalues[0]:.4f}, λ2 = {eigenvalues[1]:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo completed! Run interactive mode for custom inputs.")
    print("=" * 60)


# Main execution
if __name__ == "__main__":
    print("\nSelect mode:")
    print("1. Interactive Mode (Input your own matrices)")
    print("2. Demo Mode (View sample operations)")
    
    mode = input("\nEnter choice (1/2): ").strip()
    
    if mode == '2':
        demo_mode()
    else:
        tool = MatrixOperationsTool()
        tool.run()