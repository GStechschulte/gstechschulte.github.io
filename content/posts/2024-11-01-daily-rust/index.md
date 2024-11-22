+++
title = 'Daily Rust - Using Trait Objects That Allow for Different Types'
date = 2024-11-01
author = 'Gabriel Stechschulte'
categories = ['rust']
draft = true
+++

Trait objects...

Associated types..

You cannot use associated types with trait objects `dyn Trait` because:
1. The compiler needs to know the exact signatures of methods at compile time to create the `vtable` for dynamic dispatch.
Different implementations may have different function signatures involving the associated types.
2. When using `dyn Trait`, you must specify the associated types because the vtables for different associated type implementations
are not compatible. For example, `dyn Y<Y1 = String>` and `dyn Y<Y1 = u32>` have different vtables and cannot be used interchangeably.

```Rust
pub trait Iterator {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;
    ...
}
```

`type Item;` is an _associated type_. Each type that implements `Iterator` must specify what type of item it produces. Here's what it looks like to implement `Iterator` for a type

```Rust
// (code from the std::env standard library module)
impl Iterator for Args {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        ...
    }
    ...
}
```

`Args` returns `String` values, so the `impl` declares `type Item = String;`. However, what if we wanted...



Performance implications of `Box<Trait>` vs `enum` delegation...

## Options

Associated type where the associated type is an `enum`?


## Tradeoffs between trait objects and generics

Use a trait object if your library is designed to be extensible. If not, an `enum` may be sufficient.

There are primarily two ways to make use of traits: as trait bounds for generics or trait objects.
