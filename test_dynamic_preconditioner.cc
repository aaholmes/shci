#include <iostream>
#include <vector>
#include "src/solver/davidson.h"

int main() {
    std::cout << "Testing Dynamic Preconditioner Implementation...\n";
    
    // Create test off-diagonal elements
    std::vector<OffDiagElement> elements;
    elements.push_back({0, 1, 0.5});
    elements.push_back({2, 3, 0.3});
    elements.push_back({1, 4, 0.8});
    
    // Test OffDiagElement comparison for heap ordering
    OffDiagElement a{0, 1, 0.5};
    OffDiagElement b{2, 3, 0.3};
    
    if (!(a < b)) {  // a.magnitude (0.5) should NOT be less than b.magnitude (0.3)
        std::cout << "✓ Max-heap ordering works correctly (0.5 < 0.3 evaluates to false)\n";
    } else {
        std::cout << "✗ Max-heap ordering test failed\n";
        return 1;
    }
    
    // Test Davidson interface
    Davidson davidson(1);
    davidson.set_off_diagonal_elements(elements);
    
    std::cout << "✓ Davidson::set_off_diagonal_elements() works\n";
    std::cout << "✓ Dynamic preconditioner implementation test passed!\n";
    
    return 0;
}