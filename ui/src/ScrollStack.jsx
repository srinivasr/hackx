import React, { useEffect, useRef } from 'react';
import './ScrollStack.css';

export const ScrollStackItem = ({ children, itemClassName = '', id, style }) => (
  <div
    id={id}
    className={`scroll-stack-card ${itemClassName}`.trim()}
    style={style}
  >
    {children}
  </div>
);

const ScrollStack = ({
  children,
  className = '',
  onActiveItemChange
}) => {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const cards = container.querySelectorAll('.scroll-stack-card');
    if (!cards.length) return;

    const handleScroll = () => {
      const viewportTop = window.scrollY;
      const stickyThreshold = 180; // approximate sticky top position + header

      let currentActive = 0;
      cards.forEach((card, index) => {
        const rect = card.getBoundingClientRect();
        // If the card has reached or passed near the sticky header area
        if (rect.top <= stickyThreshold + 50) {
          currentActive = index;
        }
      });

      onActiveItemChange?.(currentActive);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    // Initial check
    handleScroll();

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [onActiveItemChange]);

  return (
    <div className={`scroll-stack-deck-container ${className}`.trim()} ref={containerRef}>
      {React.Children.map(children, (child, index) => {
        if (!React.isValidElement(child)) return child;
        return React.cloneElement(child, {
          id: `arch-step-card-${index}`,
          style: {
            ...child.props.style,
            zIndex: 10 + index
          }
        });
      })}
    </div>
  );
};

export default ScrollStack;
