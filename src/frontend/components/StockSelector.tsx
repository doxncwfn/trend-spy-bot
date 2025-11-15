import { useState } from "react";
import { Button } from "@/frontend/components/ui/button";
import { Input } from "@/frontend/components/ui/input";
import { Search, Plus } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/frontend/components/ui/select";

const POPULAR_STOCKS = [
  { symbol: "AAPL", name: "Apple Inc." },
  { symbol: "TSLA", name: "Tesla Inc." },
  { symbol: "GOOGL", name: "Alphabet Inc." },
  { symbol: "MSFT", name: "Microsoft Corp." },
  { symbol: "AMZN", name: "Amazon.com Inc." },
  { symbol: "META", name: "Meta Platforms Inc." },
  { symbol: "NVDA", name: "NVIDIA Corp." },
  { symbol: "AMD", name: "Advanced Micro Devices" },
  { symbol: "NFLX", name: "Netflix Inc." },
  { symbol: "DIS", name: "Walt Disney Co." },
];

interface StockSelectorProps {
  onSelectStock: (symbol: string) => void;
  selectedStocks: string[];
}

export const StockSelector = ({ onSelectStock, selectedStocks }: StockSelectorProps) => {
  const [customSymbol, setCustomSymbol] = useState("");
  const [selectedFromDropdown, setSelectedFromDropdown] = useState("");

  const handleAddCustomStock = () => {
    const symbol = customSymbol.toUpperCase().trim();
    if (symbol && !selectedStocks.includes(symbol)) {
      onSelectStock(symbol);
      setCustomSymbol("");
    }
  };

  const handleSelectFromDropdown = (value: string) => {
    setSelectedFromDropdown(value);
    if (value && !selectedStocks.includes(value)) {
      onSelectStock(value);
    }
  };

  return (
    <div className="flex flex-col sm:flex-row gap-4 w-full">
      <div className="flex-1">
        <Select value={selectedFromDropdown} onValueChange={handleSelectFromDropdown}>
          <SelectTrigger className="w-full bg-secondary border-border">
            <SelectValue placeholder="Select a popular stock" />
          </SelectTrigger>
          <SelectContent>
            {POPULAR_STOCKS.map((stock) => (
              <SelectItem
                key={stock.symbol}
                value={stock.symbol}
                disabled={selectedStocks.includes(stock.symbol)}
              >
                {stock.symbol} - {stock.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Enter ticker"
            value={customSymbol}
            onChange={(e) => setCustomSymbol(e.target.value.toUpperCase())}
            onKeyPress={(e) => e.key === "Enter" && handleAddCustomStock()}
            className="pl-10 bg-secondary border-border"
          />
        </div>
        <Button
          onClick={handleAddCustomStock}
          disabled={!customSymbol.trim()}
          size="icon"
          variant="default"
        >
          <Plus className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
