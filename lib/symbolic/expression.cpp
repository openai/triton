#include <cassert>
#include <vector>
#include "isaac/array.h"
#include "isaac/value_scalar.h"
#include "isaac/symbolic/expression.h"
#include "isaac/symbolic/preset.h"

namespace isaac
{

void fill(tree_node &x, invalid_node)
{
  x.subtype = INVALID_SUBTYPE;
  x.dtype = INVALID_NUMERIC_TYPE;
}

void fill(tree_node & x, std::size_t node_index)
{
  x.subtype = COMPOSITE_OPERATOR_TYPE;
  x.dtype = INVALID_NUMERIC_TYPE;
  x.node_index = node_index;
}

void fill(tree_node & x, for_idx_t index)
{
  x.subtype = FOR_LOOP_INDEX_TYPE;
  x.dtype = INVALID_NUMERIC_TYPE;
  x.for_idx = index;
}

void fill(tree_node & x, array_base const & a)
{
  x.subtype = DENSE_ARRAY_TYPE;
  x.dtype = a.dtype();
  x.array = (array_base*)&a;
}

void fill(tree_node & x, value_scalar const & v)
{
  x.dtype = v.dtype();
  x.subtype = VALUE_SCALAR_TYPE;
  x.vscalar = v.values();
}

tree_node::tree_node(){}

//
op_element::op_element() {}
op_element::op_element(operation_type_family const & _type_family, operation_type const & _type) : type_family(_type_family), type(_type){}

//
expression_tree::expression_tree(for_idx_t const &lhs, for_idx_t const &rhs, const op_element &op)
 : tree_(1), root_(0), context_(NULL), dtype_(INVALID_NUMERIC_TYPE), shape_(1)
{
  fill(tree_[0].lhs, lhs);
  tree_[0].op = op;
  fill(tree_[0].rhs, rhs);
}

expression_tree::expression_tree(for_idx_t const &lhs, value_scalar const &rhs, const op_element &op, const numeric_type &dtype)
 : tree_(1), root_(0), context_(NULL), dtype_(dtype), shape_(1)
{
  fill(tree_[0].lhs, lhs);
  tree_[0].op = op;
  fill(tree_[0].rhs, rhs);
}

expression_tree::expression_tree(value_scalar const &lhs, for_idx_t const &rhs, const op_element &op, const numeric_type &dtype)
 : tree_(1), root_(0), context_(NULL), dtype_(dtype), shape_(1)
{
  fill(tree_[0].lhs, lhs);
  tree_[0].op = op;
  fill(tree_[0].rhs, rhs);
}



//expression_tree(for_idx_t const &lhs, for_idx_t const &rhs, const op_element &op);
//expression_tree(for_idx_t const &lhs, value_scalar const &rhs, const op_element &op, const numeric_type &dtype);

template<class LT, class RT>
expression_tree::expression_tree(LT const & lhs, RT const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, shape_t const & shape) :
  tree_(1), root_(0), context_(&context), dtype_(dtype), shape_(shape)
{
  fill(tree_[0].lhs, lhs);
  tree_[0].op = op;
  fill(tree_[0].rhs, rhs);
}

template<class RT>
expression_tree::expression_tree(expression_tree const & lhs, RT const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, shape_t const & shape) :
 tree_(lhs.tree_.size() + 1), root_(tree_.size()-1), context_(&context), dtype_(dtype), shape_(shape)
{
  std::copy(lhs.tree_.begin(), lhs.tree_.end(), tree_.begin());
  fill(tree_[root_].lhs, lhs.root_);
  tree_[root_].op = op;
  fill(tree_[root_].rhs, rhs);
}

template<class LT>
expression_tree::expression_tree(LT const & lhs, expression_tree const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, shape_t const & shape) :
  tree_(rhs.tree_.size() + 1), root_(tree_.size() - 1), context_(&context), dtype_(dtype), shape_(shape)
{
  std::copy(rhs.tree_.begin(), rhs.tree_.end(), tree_.begin());
  fill(tree_[root_].lhs, lhs);
  tree_[root_].op = op;
  fill(tree_[root_].rhs, rhs.root_);
}

expression_tree::expression_tree(expression_tree const & lhs, expression_tree const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, shape_t const & shape):
  tree_(lhs.tree_.size() + rhs.tree_.size() + 1), root_(tree_.size()-1), context_(&context), dtype_(dtype), shape_(shape)
{  
  std::size_t lsize = lhs.tree_.size();
  std::copy(lhs.tree_.begin(), lhs.tree_.end(), tree_.begin());
  std::copy(rhs.tree_.begin(), rhs.tree_.end(), tree_.begin() + lsize);
  fill(tree_[root_].lhs, lhs.root_);
  tree_[root_].op = op;
  fill(tree_[root_].rhs, lsize + rhs.root_);
  for(container_type::iterator it = tree_.begin() + lsize ; it != tree_.end() - 1 ; ++it){
    if(it->lhs.subtype==COMPOSITE_OPERATOR_TYPE) it->lhs.node_index+=lsize;
    if(it->rhs.subtype==COMPOSITE_OPERATOR_TYPE) it->rhs.node_index+=lsize;
  }
  root_ = tree_.size() - 1;
}

template expression_tree::expression_tree(expression_tree const &, value_scalar const &, op_element const &,  driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(expression_tree const &, invalid_node const &, op_element const &,  driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(expression_tree const &, array_base const &,        op_element const &,  driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(expression_tree const &, for_idx_t const &,        op_element const &,  driver::Context const &, numeric_type const &, shape_t const &);

template expression_tree::expression_tree(value_scalar const &, value_scalar const &,        op_element const &, driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(value_scalar const &, invalid_node const &,        op_element const &, driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(value_scalar const &, array_base const &,        op_element const &, driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(value_scalar const &, expression_tree const &, op_element const &,  driver::Context const &, numeric_type const &, shape_t const &);

template expression_tree::expression_tree(invalid_node const &, value_scalar const &, op_element const &,  driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(invalid_node const &, expression_tree const &, op_element const &,  driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(invalid_node const &, invalid_node const &, op_element const &, driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(invalid_node const &, array_base const &,        op_element const &, driver::Context const &, numeric_type const &, shape_t const &);

template expression_tree::expression_tree(array_base const &, expression_tree const &, op_element const &,         driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(array_base const &, value_scalar const &, op_element const &, driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(array_base const &, invalid_node const &, op_element const &, driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(array_base const &, array_base const &,        op_element const &, driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(array_base const &, for_idx_t const &, op_element const &,         driver::Context const &, numeric_type const &, shape_t const &);

template expression_tree::expression_tree(for_idx_t const &, expression_tree const &, op_element const &,         driver::Context const &, numeric_type const &, shape_t const &);
template expression_tree::expression_tree(for_idx_t const &, array_base const &,        op_element const &, driver::Context const &, numeric_type const &, shape_t const &);

expression_tree::container_type & expression_tree::tree()
{ return tree_; }

expression_tree::container_type const & expression_tree::tree() const
{ return tree_; }

std::size_t expression_tree::root() const
{ return root_; }

driver::Context const & expression_tree::context() const
{ return *context_; }

numeric_type const & expression_tree::dtype() const
{ return dtype_; }

shape_t expression_tree::shape() const
{ return shape_; }

int_t expression_tree::dim() const
{ return (int_t)shape_.size(); }

expression_tree expression_tree::operator-()
{ return expression_tree(*this,  invalid_node(), op_element(UNARY_TYPE_FAMILY, SUB_TYPE), *context_, dtype_, shape_); }

expression_tree expression_tree::operator!()
{ return expression_tree(*this, invalid_node(), op_element(UNARY_TYPE_FAMILY, NEGATE_TYPE), *context_, INT_TYPE, shape_); }

//

expression_tree::node const & lhs_most(expression_tree::container_type const & array, expression_tree::node const & init)
{
  expression_tree::node const * current = &init;
  while (current->lhs.subtype==COMPOSITE_OPERATOR_TYPE)
    current = &array[current->lhs.node_index];
  return *current;
}

expression_tree::node const & lhs_most(expression_tree::container_type const & array, size_t root)
{ return lhs_most(array, array[root]); }

//
expression_tree for_idx_t::operator=(value_scalar const & r) const { return expression_tree(*this, r, op_element(BINARY_TYPE_FAMILY,ASSIGN_TYPE), r.dtype()); }
expression_tree for_idx_t::operator=(expression_tree const & r) const { return expression_tree(*this, r, op_element(BINARY_TYPE_FAMILY,ASSIGN_TYPE), r.context(), r.dtype(), r.shape()); }

expression_tree for_idx_t::operator+=(value_scalar const & r) const { return *this = *this + r; }
expression_tree for_idx_t::operator-=(value_scalar const & r) const { return *this = *this - r; }
expression_tree for_idx_t::operator*=(value_scalar const & r) const { return *this = *this * r; }
expression_tree for_idx_t::operator/=(value_scalar const & r) const { return *this = *this / r; }

}
